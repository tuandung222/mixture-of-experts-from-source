---
title: JetMoE
---

# JetMoE

JetMoE (MyShell.ai, MIT, March 2024) là model đặc biệt: Mixture of Experts không chỉ ở FFN mà còn ở **attention head**. "MoA" (Mixture of Attention heads) + "MoE" (Mixture of FFN experts) cùng layer. Đại diện cho hướng mở rộng MoE pattern qua nhiều module.

## Context

- **Tác giả**: Yikang Shen et al. (MyShell.ai, MIT, IBM).
- **Release**: March 2024.
- **Paper**: "JetMoE: Reaching Llama2 Performance with 0.1M Dollars".
- **License**: Apache 2.0.
- **Variants**: JetMoE-8B (8B total, 2.2B active).

Highlight: train với $0.1M USD (vs hàng triệu cho Llama-2). Demonstrate MoE economics.

## Config key

```python
class JetMoeConfig:
    hidden_size = 2048
    intermediate_size = 5632
    num_hidden_layers = 24
    num_attention_heads = 32
    num_key_value_heads = 32           # MHA
    num_local_experts = 8
    num_experts_per_tok = 2            # top-2 cho MoE FFN
    num_local_experts_for_attention = 8 # top-2 cho MoA attention
    num_experts_per_tok_for_attention = 2
    output_router_logits = True
    router_aux_loss_coef = 0.01
    vocab_size = 32000
```

Active params: ~2.2B / 8B total.

## Cấu trúc

```
modeling_jetmoe.py (830 dòng)
├── JetMoeMoA                       # Mixture of Attention experts
├── JetMoeMoE                       # Mixture of FFN experts
├── JetMoeAttention                 # Attention với MoA
├── JetMoeBlock                     # Decoder block
├── JetMoeTopKRouter
├── JetMoePreTrainedModel
├── JetMoeModel
└── JetMoeForCausalLM
```

## Concept: Mixture of Attention (MoA)

Idea: thay vì có 32 attention heads, có **8 attention experts**, mỗi expert là một group attention heads (4 heads/expert). Router chọn top-2 expert (8 heads) cho mỗi token.

So với MHA:

- MHA: mọi token đi qua tất cả 32 head.
- MoA: mỗi token đi qua 8 head (top-2 expert × 4 head).

Active attention compute giảm 4x. Quality giữ vì 32 head trở thành "pool" 8 expert specialize.

## `JetMoeMoA`

```python
class JetMoeMoA(nn.Module):
    """Mixture of Attention Experts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts_for_attention
        self.top_k = config.num_experts_per_tok_for_attention
        self.hidden_size = config.hidden_size
        self.head_dim = config.head_dim
        # Heads per expert
        self.num_heads_per_expert = config.num_attention_heads // self.num_experts

        # Expert weights: (num_experts, hidden, heads_per_expert * head_dim * 3)
        # For Q, K, V combined
        self.experts = nn.Parameter(...)
        self.router = JetMoeTopKRouter(config, target="attention")
```

(Pseudocode dựa trên `src/transformers/models/jetmoe/modeling_jetmoe.py`.)

Forward concept:

1. Router score expert dựa trên hidden_states.
2. Top-k expert selected.
3. Gather expert weight tương ứng.
4. Compute Q, K, V chỉ cho selected experts.
5. Attention với reduced number of heads.
6. Combine output via routing weight.

Code thực phức tạp vì cần merge Q, K, V cho top-k expert thành tensor liền nhau cho `scaled_dot_product_attention`.

## `JetMoeMoE` (FFN MoE)

```python
class JetMoeMoE(nn.Module):
    """Mixture of FFN Experts (standard MoE)."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_tok
        ...
        self.gate_up_proj = nn.Parameter(...)
        self.down_proj = nn.Parameter(...)

    def forward(self, hidden_states, routing_indices, routing_weights):
        # Standard MoE FFN forward
        ...
```

Standard MoE pattern giống Mixtral cho FFN.

## `JetMoeBlock`

```python
class JetMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_layernorm = JetMoeRMSNorm(config.hidden_size)
        self.self_attn = JetMoeAttention(config)
        self.post_attention_layernorm = JetMoeRMSNorm(config.hidden_size)
        self.mlp = JetMoeMoE(config)

    def forward(self, hidden_states, ...):
        # Attention with MoA (inside self.self_attn)
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, attn_router_logits, ... = self.self_attn(hidden_states, ...)
        hidden_states = residual + attn_output

        # FFN with MoE
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, mlp_router_logits = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states, attn_router_logits, mlp_router_logits, ...
```

(Pseudocode pattern.)

Layer có **hai router**: một cho attention MoA, một cho FFN MoE. Hai router logits trả riêng để compute aux loss cho mỗi.

## Aux loss kép

```python
class JetMoeForCausalLM(...):
    def forward(self, ..., output_router_logits=None, ...):
        outputs = self.model(...)
        ...
        if labels is not None:
            loss = self.loss_function(logits, labels)
            if output_router_logits:
                aux_loss_attn = load_balancing_loss_func(
                    outputs.attn_router_logits,
                    num_experts=self.config.num_local_experts_for_attention,
                    top_k=self.config.num_experts_per_tok_for_attention,
                )
                aux_loss_mlp = load_balancing_loss_func(
                    outputs.mlp_router_logits,
                    num_experts=self.config.num_local_experts,
                    top_k=self.config.num_experts_per_tok,
                )
                loss += self.router_aux_loss_coef * (aux_loss_attn + aux_loss_mlp)
        ...
```

Tổng aux loss cộng từ MoA và MoE. Cùng coef (default 0.01).

## Vì sao có MoA

Bài toán: MHA expensive ở context dài. KV cache lớn. Compute `Q @ K^T` quadratic theo seq.

MoA giải:

1. **Reduce compute**: chỉ top-k expert active → 1/4 attention compute.
2. **Specialization**: expert chuyên pattern khác nhau (gần token, far token, semantic, syntactic).
3. **KV cache giảm**: chỉ store K/V của heads active mỗi token. Hard implementation (KV cache phải biết token nào đã active expert nào).

Trade-off:

1. **Implementation phức tạp**. Cache, dispatch, mask đều cần custom.
2. **Quality drop trong một số benchmark**. Long-range attention bị giới hạn nếu chỉ active 8/32 heads.
3. **Ít được adopt**. JetMoE là chính, một số research extension. Mainstream LLM 2024-2025 không follow.

## Pitfall

**1. MoA cache layout**: KV cache shape phụ thuộc routing. Per-token routing có thể khác nhau, cache không đồng nhất. JetMoE giải bằng cache full (mọi head store) nhưng chỉ active subset → memory không tiết kiệm.

**2. MoA backward pass**: gradient phải flow qua router của attention. Complex.

**3. Aux loss balance giữa MoA và MoE**: nếu coef cùng nhưng scale khác, một loss dominate. Cần tune riêng.

**4. JetMoE generate**: với MoA, cache management complex hơn. Default `generate` API trong HF có thể không tối ưu.

**5. Confuse MoA với GQA**: GQA reduce KV heads (1/4 K/V vs Q). MoA reduce active attention compute (1/4 heads per token). Khác concept.

Chương sau ta đọc Jamba.
