---
title: Qwen3-MoE
---

# Qwen3-MoE

Qwen3-MoE (Alibaba/Qwen team, 2024-2025) là dòng MoE model với infrastructure modern: `grouped_mm`, `@use_experts_implementation` chuẩn HF, không có gimmick đặc biệt. Đại diện cho "production-ready, no surprises" design. Đọc nhanh sau Mixtral và DeepSeek vì pattern quen.

## Context

- **Tác giả**: Qwen team (Alibaba Cloud).
- **Release**: 2024-2025.
- **Variants**: Qwen3-30B-A3B (30B total, 3B active), Qwen3-235B-A22B (235B total, 22B active).
- **License**: Apache 2.0.
- **Paper**: "Qwen3 Technical Report".

## Config key

Qwen3-30B-A3B:

```python
class Qwen3MoeConfig:
    hidden_size = 2048
    intermediate_size = 6144           # cho dense
    moe_intermediate_size = 768          # cho mỗi routed expert
    num_hidden_layers = 48
    num_attention_heads = 32
    num_key_value_heads = 4              # GQA 8:1
    num_experts = 128
    num_experts_per_tok = 8              # top-k
    decoder_sparse_step = 1              # mọi layer là MoE (giống Mixtral)
    norm_topk_prob = False               # KHÔNG renormalize
    output_router_logits = False
    router_aux_loss_coef = 0.001
    vocab_size = 151936
```

Qwen3-235B-A22B:

```python
hidden_size = 4096
intermediate_size = 12288
moe_intermediate_size = 1536
num_hidden_layers = 94
num_experts = 128
num_experts_per_tok = 8
```

Note: `norm_topk_prob = False`. Khác Mixtral và DeepSeek-V3 (both True).

## Cấu trúc

```
modeling_qwen3_moe.py (733 dòng)
├── Qwen3MoeAttention                # GQA + RoPE
├── Qwen3MoeMLP                      # Dense MLP
├── Qwen3MoeExperts                  # 3D weight, @use_experts_implementation
├── Qwen3MoeTopKRouter               # Linear + softmax + topk
├── Qwen3MoeSparseMoeBlock           # Wrap router + experts
├── Qwen3MoeDecoderLayer
├── Qwen3MoePreTrainedModel
├── Qwen3MoeRotaryEmbedding
├── Qwen3MoeRMSNorm
├── Qwen3MoeModel
└── Qwen3MoeForCausalLM
```

## `Qwen3MoeExperts`

```python
@use_experts_implementation
class Qwen3MoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager loop, giống MixtralExperts
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```

(`src/transformers/models/qwen3_moe/modeling_qwen3_moe.py`, class `Qwen3MoeExperts`.)

**Identical** với MixtralExperts. Naming khác (`num_experts` thay vì `num_local_experts`, `moe_intermediate_size` thay vì `intermediate_size`), nhưng logic giống đúc.

## `Qwen3MoeTopKRouter`

```python
class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.norm_topk_prob = config.norm_topk_prob
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

(class `Qwen3MoeTopKRouter`.)

Khác Mixtral 1 detail: `if self.norm_topk_prob` (optional). Default `False`.

**Tại sao không renormalize?**

Argument từ Qwen team: với top-k cao (k=8), top-k weights tổng đã gần 1 (softmax tập trung ở top expert). Renormalize có thể méo distribution, không cần thiết.

Empirical: cả normalize và không đều work. Qwen chọn không. DeepSeek chọn có (vì sigmoid, không có sum=1 inherent).

## `Qwen3MoeSparseMoeBlock`

```python
class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig):
        super().__init__()
        self.experts = Qwen3MoeExperts(config)
        self.gate = Qwen3MoeTopKRouter(config)

    def forward(self, hidden_states):
        batch_size, seq_len, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, seq_len, hidden_dim)
        return hidden_states, router_logits
```

(class `Qwen3MoeSparseMoeBlock`.)

Khác Mixtral:

1. **Không jitter noise**. Qwen3 không có `router_jitter_noise` trong config.
2. **Trả `router_logits` ra ngoài**. Để aux loss compute. Mixtral dùng `OutputRecorder` capture; Qwen3 explicit return.

## `Qwen3MoeDecoderLayer`

```python
class Qwen3MoeDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = Qwen3MoeAttention(config, layer_idx)
        # Mọi layer là MoE
        if (layer_idx not in config.mlp_only_layers) and (
            config.num_experts > 0 and (layer_idx + 1) % config.decoder_sparse_step == 0
        ):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(config, intermediate_size=config.intermediate_size)

        self.input_layernorm = Qwen3MoeRMSNorm(...)
        self.post_attention_layernorm = Qwen3MoeRMSNorm(...)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, ... = self.self_attn(hidden_states, ...)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output = self.mlp(hidden_states)

        if isinstance(mlp_output, tuple):
            hidden_states_out, router_logits = mlp_output
        else:
            hidden_states_out = mlp_output
            router_logits = None

        hidden_states = residual + hidden_states_out
        return hidden_states, router_logits, ...
```

(class `Qwen3MoeDecoderLayer`.)

Phân tích `if` condition:

1. `layer_idx not in config.mlp_only_layers`: nếu user chỉ định layer này là dense (legacy), không dùng MoE.
2. `num_experts > 0 and (layer_idx + 1) % decoder_sparse_step == 0`: cho phép alternate sparse/dense theo step.

Default `decoder_sparse_step = 1` và `mlp_only_layers = []`: mọi layer MoE.

Tương tự DeepSeek (3 layer đầu dense), Qwen3 cho phép config flexibility.

## `Qwen3MoeAttention`

```python
@use_kernelized_func(apply_rotary_pos_emb)
class Qwen3MoeAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
        self.o_proj = nn.Linear(...)
        self.q_norm = Qwen3MoeRMSNorm(self.head_dim, ...)
        self.k_norm = Qwen3MoeRMSNorm(self.head_dim, ...)
```

(class `Qwen3MoeAttention`.)

Đặc thù Qwen3: **QK norm** (RMSNorm trên Q và K trước attention). Stabilize training với context dài. Tương tự Llama-3.1/3.2.

GQA 8:1 (32 head Q, 4 head KV).

## `Qwen3MoeForCausalLM`

```python
class Qwen3MoeForCausalLM(Qwen3MoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_allgather"}
    _sp_plan = {"lm_head": "colwise_loss_parallel"}

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3MoeModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.post_init()
```

(class `Qwen3MoeForCausalLM`.)

`_sp_plan` (sequence parallel plan) khai báo. Hỗ trợ TP + SP cho long context.

## Modular file

Qwen3-MoE có `modular_qwen3_moe.py` thừa kế từ Mixtral. Modeling file là generated, header có warning không edit thủ công.

Modular structure (simplified):

```python
# modular_qwen3_moe.py
from ..mixtral.modeling_mixtral import MixtralAttention, MixtralExperts, ...

class Qwen3MoeAttention(MixtralAttention):
    # Override QK norm
    ...

class Qwen3MoeExperts(MixtralExperts):
    pass  # Identical, chỉ rename

class Qwen3MoeTopKRouter(MixtralTopKRouter):
    # Override để add norm_topk_prob flag
    ...
```

Khi đọc Qwen3 modeling, thực ra đọc Mixtral với vài tweak.

## Khác biệt nhỏ so với Mixtral

| Aspect | Mixtral | Qwen3-MoE |
|---|---|---|
| Top-k | 2 | 8 |
| Num experts | 8 | 128 |
| Expert FFN size | 14336 (1.75x dense) | 768-1536 (small) |
| Shared expert | Không | Không |
| Renormalize topk | Có | Không (default) |
| QK norm | Không | Có |
| Jitter noise | Có (default 0) | Không |
| Aux loss | 0.001 | 0.001 |
| Sparse/dense alternate | Không | Configurable |

Qwen3 nhìn chung là "Mixtral fine-grained" + QK norm.

## Pitfall

**1. Active params nhầm**: Qwen3-30B-A3B nghĩa là "30B total, A3B active". A = active. Không phải 30B * 3 = 90B.

**2. `norm_topk_prob=False`**: nếu fork để train, có thể quên flag này. Norm vs no norm cho output magnitude khác.

**3. QK norm dtype**: `q_norm`, `k_norm` apply trước RoPE. Dtype có thể bị cast khác nhau, gây sai số.

**4. `decoder_sparse_step`**: default 1 (mọi layer MoE). Nếu set 2 (xen kẽ), active params giảm 1/2 vì half layer là dense.

**5. EP với 128 expert**: phải GPU số chia hết 128 hoặc compatible factor (8, 16, 32, 64).

Chương sau ta đọc GPT-OSS.
