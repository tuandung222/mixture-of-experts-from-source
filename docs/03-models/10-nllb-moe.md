---
title: NLLB-MoE
---

# NLLB-MoE

NLLB-MoE (Meta AI, July 2022) là model translation MoE đầu tiên ở scale production. NLLB = "No Language Left Behind", target 200 ngôn ngữ. 54B total params với MoE FFN trong both encoder và decoder. Đại diện cho translation paradigm với expert dropout regularization.

## Context

- **Tác giả**: Meta AI (NLLB Team).
- **Release**: July 2022.
- **Paper**: "No Language Left Behind: Scaling Human-Centered Machine Translation" (NLLB Team, 2022).
- **License**: CC-BY-NC 4.0.
- **Variants**: NLLB-MoE-54B.
- **Architecture**: encoder-decoder (Transformer style).

## Config key

```python
class NllbMoeConfig:
    d_model = 2048
    encoder_layers = 24
    decoder_layers = 24
    encoder_ffn_dim = 8192
    decoder_ffn_dim = 8192
    encoder_attention_heads = 16
    decoder_attention_heads = 16
    num_experts = 128
    expert_capacity = 64                # capacity per expert
    encoder_sparse_step = 4              # MoE mỗi 4 layer encoder
    decoder_sparse_step = 4              # MoE mỗi 4 layer decoder
    router_aux_loss_coef = 0.01
    router_z_loss_coef = 0.001
    router_bias = False
    router_ignore_padding_tokens = True
    router_jitter_noise = 0.01
    router_dtype = "float32"
    moe_token_dropout = 0.2               # Expert dropout
    moe_eval_capacity_token_fraction = 1.0
    vocab_size = 256206                   # large vocab for 200 languages
```

Note đặc thù:

- `moe_token_dropout = 0.2`: drop ngẫu nhiên 20% expert ở training.
- `expert_capacity = 64`: capacity-based, drop token nếu vượt.
- `encoder_sparse_step = 4`: chỉ layer 0, 4, 8, ..., 20 là MoE (6 trong 24 layer).

## Cấu trúc

```
modeling_nllb_moe.py (1143 dòng)
├── NllbMoeAttention                # Standard MHA
├── NllbMoeDenseActDense             # Dense FFN (cho non-MoE layer)
├── NllbMoeTop2Router                # Top-2 router với capacity
├── NllbMoeSparseMLP                 # Wrap router + experts + dropout
├── NllbMoeLayer                     # Block với selectable sparse/dense
├── NllbMoeEncoderLayer              # Encoder layer
├── NllbMoeDecoderLayer              # Decoder layer (causal + cross-attn)
├── NllbMoeEncoder
├── NllbMoeDecoder
├── NllbMoeModel
└── NllbMoeForConditionalGeneration  # Translation head
```

## Top-2 router với capacity

```python
class NllbMoeTop2Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def forward(self, hidden_states, padding_mask=None):
        ...
        router_logits = self.classifier(hidden_states.to(self.dtype))
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype)

        # Top-2 selection
        top_2_values, top_2_indices = torch.topk(router_probs, 2, dim=-1)
        top_2_values = top_2_values / (top_2_values.sum(dim=-1, keepdim=True) + 1e-7)

        # Build expert mask and apply capacity
        # ... (logic giống Switch nhưng cho top-2)
```

(Pseudocode dựa trên `src/transformers/models/nllb_moe/modeling_nllb_moe.py`, class `NllbMoeTop2Router`.)

Khác Switch:

1. **Top-2** thay vì top-1.
2. **Capacity vẫn áp dụng** với top-2: 2 token có thể đến cùng expert, cả hai counted.
3. **Token capacity factor configurable** ở inference.

## Expert dropout

```python
class NllbMoeSparseMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = NllbMoeTop2Router(config)
        self.experts = nn.ModuleDict({
            f"expert_{i}": NllbMoeDenseActDense(config)
            for i in range(config.num_experts)
        })
        self.token_dropout = config.moe_token_dropout

    def forward(self, hidden_states, padding_mask=None):
        ...
        if self.training:
            # Drop entire expert with probability token_dropout
            ...
        ...
```

(Pseudocode.)

**Expert dropout**: ở training, mỗi forward pass, ~20% expert bị "drop" (output forced to 0). Khác standard dropout:

- Standard dropout: drop **token activations** (~10-15%).
- Expert dropout: drop **toàn bộ expert** (entire MLP).

Regularization mạnh. Force router không over-rely vào subset expert. Ngăn collapse.

## Encoder-decoder integration

```python
class NllbMoeEncoderLayer(nn.Module):
    def __init__(self, config, is_sparse=False):
        super().__init__()
        self.self_attn = NllbMoeAttention(config)
        self.self_attn_layer_norm = ...
        if is_sparse:
            self.ffn = NllbMoeSparseMLP(config)
        else:
            self.ffn = NllbMoeDenseActDense(config)
        self.ffn_layer_norm = ...

class NllbMoeDecoderLayer(nn.Module):
    def __init__(self, config, is_sparse=False):
        super().__init__()
        self.self_attn = NllbMoeAttention(config)  # causal
        self.encoder_attn = NllbMoeAttention(config)  # cross
        self.encoder_attn_layer_norm = ...
        if is_sparse:
            self.ffn = NllbMoeSparseMLP(config)
        else:
            self.ffn = NllbMoeDenseActDense(config)
```

Encoder layer: self-attention + FFN.
Decoder layer: self-attention (causal) + cross-attention (to encoder) + FFN.

FFN có thể sparse hoặc dense theo index.

## `encoder_sparse_step`

```python
class NllbMoeEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.ModuleList([
            NllbMoeEncoderLayer(config, is_sparse=(i % config.encoder_sparse_step == 0))
            for i in range(config.encoder_layers)
        ])
```

Với `encoder_sparse_step=4`: layer 0, 4, 8, 12, 16, 20 là sparse (6/24). Còn lại dense.

Tương tự decoder.

## Vì sao alternate sparse/dense

NLLB design choice:

1. **Compute saving**: dense layer rẻ hơn MoE (no routing overhead). Mix giúp giảm latency.
2. **Stabilization**: dense layer học "common" pattern, MoE chuyên hoá. Pattern xen kẽ ổn định training.
3. **Memory**: 6 MoE layer × 128 expert thay vì 24 layer × 128. Total params giảm.

Switch dùng pattern này. NLLB tiếp tục.

## Translation-specific aspects

NLLB là **machine translation** model. Đặc thù:

**1. Source-target language tagging**: input có prefix `<source_lang>` và `<target_lang>` token. Router có thể học specialize theo language pair.

**2. Vocab lớn**: 256k tokens. Để cover 200 ngôn ngữ. So với LLM English (32k-128k), khá lớn.

**3. Beam search default**: translation thường beam search (num_beams=5). Aux loss không apply ở generate.

**4. Length penalty + early stopping**: translation length predict mạnh, có heuristic stop.

## Aux loss và Z-loss

```python
class NllbMoeForConditionalGeneration(...):
    def forward(self, ...):
        ...
        if labels is not None:
            loss = ce_loss
            if output_router_logits:
                aux_loss = ...
                z_loss = ...
                loss = loss + self.config.router_aux_loss_coef * aux_loss
                loss = loss + self.config.router_z_loss_coef * z_loss
```

Coef aux = 0.01, z = 0.001. Switch-style.

## So sánh NLLB với Switch

| Aspect | Switch | NLLB-MoE |
|---|---|---|
| Year | 2021 | 2022 |
| Task | T5-style (any text-to-text) | Translation specific |
| Top-k | 1 | 2 |
| Capacity | Yes (factor 1.0-1.25) | Yes (capacity 64) |
| Expert dropout | No | Yes (20%) |
| Encoder-decoder | Yes | Yes |
| Sparse step | Configurable | 4 |

NLLB là evolution của Switch cho translation.

## Pitfall

**1. Capacity vs num_experts**: capacity 64 với 128 expert nghĩa là **trung bình** 64 token per expert. Với batch nhỏ, ratio sai. Tune theo workload.

**2. Expert dropout ở inference**: phải tắt. `if self.training` check.

**3. Source-target language pair**: nếu fine-tune sang pair khác, có thể cần warm-up router.

**4. Encoder-decoder cache**: K/V cross-attention compute từ encoder, không recompute mỗi decode step. Standard encoder-decoder pattern.

**5. ModuleDict scaling**: 128 expert × 24 layer * 2 (enc+dec) trong nhưng chỉ 6+6 sparse → 12 layer × 128 = 1536 expert modules. Loading slow.

Chương sau ta đọc PhiMoE.
