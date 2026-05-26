---
title: PhiMoE
---

# PhiMoE

PhiMoE (Microsoft Research, August 2024) là MoE version của Phi-3.5. Microsoft target small-scale efficient model. Phi-3.5-MoE-instruct là chính. Pattern code rất gần Mixtral nhưng config nhỏ hơn nhiều. Đại diện cho "small-MoE for edge deployment".

## Context

- **Tác giả**: Microsoft Research.
- **Release**: August 2024.
- **Paper**: "Phi-3 Technical Report" (extended with MoE variant).
- **License**: MIT.
- **Variants**: Phi-3.5-MoE-instruct (42B total, 6.6B active).

## Config key

Phi-3.5-MoE:

```python
class PhimoeConfig:
    hidden_size = 4096
    intermediate_size = 6400
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8           # GQA 4:1
    num_local_experts = 16
    num_experts_per_tok = 2
    router_jitter_noise = 0.01
    router_aux_loss_coef = 0.001
    rope_theta = 10000.0
    sliding_window = None
    attention_bias = True             # Phi style
    lm_head_bias = True
    vocab_size = 32064
```

Active per token: ~6.6B / 42B total.

## Cấu trúc

```
modeling_phimoe.py (918 dòng)
├── PhimoeRMSNorm
├── PhimoeRotaryEmbedding
├── PhimoeAttention                  # GQA with QK bias
├── PhimoeMLP                         # Dense MLP
├── PhimoeTopKRouter                 # Linear + softmax + topk
├── PhimoeExperts                    # ModuleList[MLP] hoặc 3D
├── PhimoeSparseMoeBlock
├── PhimoeDecoderLayer
├── PhimoePreTrainedModel
├── PhimoeModel
├── PhimoeForCausalLM
├── PhimoeForSequenceClassification
└── load_balancing_loss_func
```

## Đặc thù PhiMoE

PhiMoE gần Mixtral nhất trong dòng "supplementary". Khác biệt:

**1. Number of experts 16, top-k 2**. Giữa Mixtral (8/2) và Qwen3 (128/8). "Sweet spot" cho small-scale.

**2. Bias trong attention** (`attention_bias=True`). Phi-3 design choice.

**3. `lm_head_bias=True`**. Cũng có bias. Khác Llama/Mixtral (no bias).

**4. Sparsemax routing experimental** (một variant của Phi-3.5-MoE). Có thể dùng sparsemax thay softmax cho sparser distribution.

## `PhimoeTopKRouter`

```python
class PhimoeTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value, router_indices
```

(`src/transformers/models/phimoe/modeling_phimoe.py`, class `PhimoeTopKRouter`.)

**Identical với MixtralTopKRouter**. Mixtral-style với renormalize.

## `PhimoeExperts`

Implementation tương tự Mixtral với 3D weight tensor và `@use_experts_implementation`.

```python
@use_experts_implementation
class PhimoeExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager loop identical to MixtralExperts
        ...
```

## `PhimoeAttention`

```python
class PhimoeAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_proj = nn.Linear(hidden_size, num_heads * head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(num_heads * head_dim, hidden_size, bias=config.attention_bias)
```

(class `PhimoeAttention`.)

QKVO có bias. Khác Mixtral (no bias).

GQA 4:1. RoPE standard.

## `PhimoeForCausalLM`

```python
class PhimoeForCausalLM(PhimoePreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = PhimoeModel(config)
        # Note: lm_head có bias
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.lm_head_bias)
        ...
```

`lm_head` có bias. Khi tied với embed_tokens, bias không tied (embed không có bias).

## So sánh PhiMoE với Mixtral

| Aspect | Mixtral 8x7B | PhiMoE |
|---|---|---|
| Total params | 46.7B | 42B |
| Active params | 12.9B | 6.6B |
| num_experts | 8 | 16 |
| top_k | 2 | 2 |
| Hidden size | 4096 | 4096 |
| Intermediate | 14336 (large) | 6400 (medium) |
| GQA | 4:1 | 4:1 |
| Attention bias | No | Yes |
| Sliding window | No | No |
| Layers | 32 | 32 |

PhiMoE smaller active params mặc dù hidden cùng. Vì:

1. Intermediate FFN nhỏ hơn (6400 vs 14336).
2. Mỗi expert nhỏ hơn.
3. Top-k cùng 2 nhưng expert size khác.

## Target use case

Microsoft target:

1. **Edge deployment**: 6.6B active fit trên A10 hoặc consumer GPU.
2. **Strong instruct following**: training với strong filtered data, supervised + RLHF.
3. **Cost-effective serving**: total params manageable trong setup nhỏ.

PhiMoE không cố compete với DeepSeek-V3 hay GPT-OSS-120B về capability. Niche: small efficient MoE.

## Pitfall

**1. Confuse Phi-3.5-MoE và Phi-3.5-MoE-instruct**: cùng base, khác fine-tune (post-training).

**2. Attention bias gặp issues với quantization**: bias là một row extra per linear. Quantize phải handle.

**3. Mixing với Phi-3 (dense)**: cùng tokenizer, cùng vocab. Có thể swap weight với care.

**4. Edge deployment**: 6.6B active vẫn không fit edge thực sự (Raspberry Pi). Cần ít nhất 8GB GPU. Marketing dùng "edge" loosely.

**5. PhiMoE và sliding window**: config có flag nhưng default None. Một số variant có sliding window 4096 cho long context.

Chương sau là tổng kết so sánh.
