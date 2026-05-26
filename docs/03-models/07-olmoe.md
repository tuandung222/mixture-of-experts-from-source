---
title: OLMoE
---

# OLMoE

OLMoE (Allen Institute for AI, September 2024) là MoE model open-data đầu tiên: training data, training code, intermediate checkpoint, recipe đều công khai. Đại diện cho academic / reproducibility-focused MoE. Pattern code gần Qwen3-MoE.

## Context

- **Tác giả**: Allen Institute for AI (AI2).
- **Release**: September 2024.
- **Paper**: "OLMoE: Open Mixture-of-Experts Language Models" (Muennighoff et al., 2024).
- **License**: Apache 2.0.
- **Variants**: OLMoE-1B-7B (1B active, 7B total).

## Config key

```python
class OlmoeConfig:
    hidden_size = 2048
    intermediate_size = 1024            # mỗi expert FFN
    num_hidden_layers = 16
    num_attention_heads = 16
    num_key_value_heads = 16             # MHA, không GQA
    num_experts = 64
    num_experts_per_tok = 8              # top-8
    norm_topk_prob = True
    output_router_logits = False
    router_aux_loss_coef = 0.01           # cao hơn Mixtral 10x
    z_loss_coef = 0.01
    rope_theta = 10000
    vocab_size = 50304
```

Active per token: ~1.3B. Total 6.9B.

## Cấu trúc

```
modeling_olmoe.py (710 dòng)
├── OlmoeMLP
├── OlmoeRMSNorm
├── OlmoeRotaryEmbedding
├── OlmoeAttention                  # MHA with QK norm
├── OlmoeTopKRouter                 # Linear + softmax + topk + renormalize
├── OlmoeExperts                    # 3D weight, @use_experts_implementation
├── OlmoeSparseMoeBlock
├── OlmoeDecoderLayer
├── OlmoePreTrainedModel
├── OlmoeModel
├── OlmoeForCausalLM
└── load_balancing_loss_func
```

## Đặc thù OLMoE

OLMoE gần Mixtral và Qwen3-MoE. Khác biệt nhỏ:

**1. Top-k cao (8) với num_experts vừa (64)**. Tỉ lệ k/E = 12.5%. Higher specialization tradeoff.

**2. Aux loss coef cao (0.01)**. So với Mixtral 0.001. Lý do paper: với 64 expert, balance khó hơn, cần coef mạnh.

**3. Z-loss enabled** (Switch-style). Coef 0.01. Stabilize router logit magnitude.

**4. MHA thay vì GQA**. Hidden 2048 nhỏ, MHA acceptable. Lớn hơn (10B+) thì GQA cần thiết hơn.

**5. Reproducible**. AI2 release dataset (Dolma), training code (OLMo repo), eval suite (OLMES). Mỗi step intermediate có checkpoint.

## `OlmoeExperts`

```python
@use_experts_implementation
class OlmoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager loop identical to Mixtral / Qwen3
        ...
```

(`src/transformers/models/olmoe/modeling_olmoe.py`, class `OlmoeExperts`.)

Identical với MixtralExperts/Qwen3MoeExperts. Naming cùng standard. Đây là dấu hiệu codebase HF tiến tới convergence.

## `OlmoeTopKRouter`

```python
class OlmoeTopKRouter(nn.Module):
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

(class `OlmoeTopKRouter`.)

Identical với Qwen3MoeTopKRouter. Cả hai dùng `norm_topk_prob` flag. OLMoE default `True` (renormalize), Qwen3 default `False`.

## `OlmoeAttention`

```python
class OlmoeAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_proj = nn.Linear(...)
        self.k_proj = nn.Linear(...)
        self.v_proj = nn.Linear(...)
        self.o_proj = nn.Linear(...)
        # QK normalize (Llama-3 / Qwen3 style)
        self.q_norm = OlmoeRMSNorm(self.head_dim)
        self.k_norm = OlmoeRMSNorm(self.head_dim)
```

QK norm có (tương tự Qwen3). MHA full.

## Z-loss implementation

OLMoE có z_loss riêng:

```python
def router_z_loss_func(router_logits):
    """Compute router z-loss."""
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = (log_z ** 2).mean()
    return z_loss
```

Trong `OlmoeForCausalLM.forward`:

```python
if labels is not None:
    loss = self.loss_function(logits, labels, ...)
    if output_router_logits:
        aux_loss = load_balancing_loss_func(outputs.router_logits, ...)
        z_loss = router_z_loss_func(outputs.router_logits)
        loss = loss + self.config.router_aux_loss_coef * aux_loss
        loss = loss + self.config.z_loss_coef * z_loss
```

Loss tổng = ce + 0.01 * aux + 0.01 * z. Đã đi sâu z-loss ở Phần 1 Chương 4.

## Open-data implications

OLMoE chứng minh:

1. MoE có thể train từ scratch với open data (Dolma 5T tokens).
2. Recipe Mixtral-style (top-2 with 8 expert) không phải optimal cho mọi scale; top-8 với 64 expert tốt hơn ở 7B.
3. Aux loss coef cần tune theo scale.

Bài học practical: nếu fork một MoE config, đừng copy aux_loss_coef cứng. Tune theo data và scale.

## Pitfall

**1. Aux coef 0.01 quá lớn cho fine-tune**: nếu user fine-tune từ pre-trained checkpoint với coef 0.01, aux loss vẫn dominate gradient. Có thể giảm xuống 0.001 cho fine-tune.

**2. Z-loss coef cùng aux coef**: nếu tăng aux mà quên tăng z, balance bị mất.

**3. MHA + 8 expert routing**: với hidden=2048 nhỏ, attention compute nhỏ, MoE routing dominate. Latency phụ thuộc routing chứ không phải attention.

**4. Top-8 trong 64 expert**: với batch nhỏ, mỗi expert chỉ thấy 1-2 token. Aux loss noisy.

**5. Open recipe có pitfall scale**: data + recipe public không guarantee model lớn (>10B) sẽ work. Mỗi scale có challenge riêng (parallelism, hyperparameter).

Chương sau ta đọc JetMoE.
