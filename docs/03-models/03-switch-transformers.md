---
title: Switch Transformers
---

# Switch Transformers

Switch Transformer (Google, 2021) là model lớn đầu tiên prove MoE ở scale T-params. Khác Mixtral và mọi LLM MoE 2024+ ở vài điểm cơ bản: encoder-decoder (T5-based), top-1 routing, expert capacity + token dropping. Là đại diện paradigm "Switch" gần như tuyệt chủng nhưng concept vẫn quan trọng.

## Context

- **Tác giả**: Fedus, Zoph, Shazeer (Google Brain).
- **Release**: January 2021 (paper); checkpoints sau đó.
- **Paper**: "Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity".
- **Base**: T5 (encoder-decoder).
- **Variant**: switch-base-8, switch-base-32, switch-base-128, switch-base-256, switch-large-128, switch-c-2048.

## Config key

Switch-Base-128:

```python
class SwitchTransformersConfig:
    d_model = 768                  # hidden_size
    d_ff = 2048
    num_layers = 12                # encoder layers
    num_decoder_layers = 12
    num_heads = 12
    num_experts = 128
    expert_capacity = 64           # capacity per expert per batch
    router_bias = False
    router_jitter_noise = 0.01
    router_dtype = "float32"       # fp32 cho router
    router_ignore_padding_tokens = False
    router_z_loss_coef = 0.001
    router_aux_loss_coef = 0.001
    vocab_size = 32128
    ...
```

Switch-C-2048 (model lớn nhất, ~1.6T params):

```python
num_experts = 2048
num_layers = 15
d_model = 2080
d_ff = 6144
```

## Cấu trúc

```
modeling_switch_transformers.py (1095 dòng, file lớn)
├── SwitchTransformersTop1Router         # Top-1 + capacity + token drop
├── SwitchTransformersLayerNorm           # T5 LayerNorm
├── SwitchTransformersDenseActDense       # Standard FFN (cho dense layer)
├── SwitchTransformersExperts             # ModuleDict (1 MLP per expert)
├── SwitchTransformersSparseMLP           # Wrap router + experts
├── SwitchTransformersLayerFF             # Forward wrapper sparse/dense
├── SwitchTransformersAttention           # T5 attention (relative position bias)
├── SwitchTransformersLayerSelfAttention
├── SwitchTransformersLayerCrossAttention # Cho decoder
├── SwitchTransformersBlock               # Encoder block
├── SwitchTransformersPreTrainedModel
├── SwitchTransformersStack               # Stack of blocks
├── SwitchTransformersModel               # Encoder + Decoder
├── SwitchTransformersForConditionalGeneration
└── SwitchTransformersEncoderModel
```

Khác Mixtral chủ yếu:

1. Encoder-decoder thay decoder-only.
2. Top-1 router với capacity.
3. Experts là `ModuleDict` (1 nn.Module mỗi expert) thay vì 3D weight tensor.
4. T5-style attention với relative position bias.

## `SwitchTransformersTop1Router`

```python
class SwitchTransformersTop1Router(nn.Module):
    """Router using tokens choose top-1 experts assignment."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        self.classifier = nn.Linear(config.hidden_size, self.num_experts, bias=config.router_bias)
        self.jitter_noise = config.router_jitter_noise
        self.ignore_padding_tokens = config.router_ignore_padding_tokens
        self.dtype = getattr(torch, config.router_dtype)

    def forward(self, hidden_states):
        # Float32 for stability
        self.input_dtype = hidden_states.dtype
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        self.classifier = self.classifier.to(self.dtype)
        router_logits = self.classifier(hidden_states)

        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        router_logits, expert_index = torch.max(router_probs, dim=-1, keepdim=True)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
        token_priority = torch.cumsum(expert_index, dim=-2)
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return router_probs, expert_index, router_logits
```

(`src/transformers/models/switch_transformers/modeling_switch_transformers.py`, class `SwitchTransformersTop1Router`.)

Phân tích:

**Step 1: Jitter noise** giống Mixtral, nhưng default 0.01 (Mixtral 0.0).

**Step 2: Cast router weight fp32**:

```python
self.classifier = self.classifier.to(self.dtype)
router_logits = self.classifier(hidden_states)
```

Khác Mixtral chỉ cast softmax fp32. Switch cast cả weight về fp32 trước linear. Conservative hơn.

**Step 3: Top-1 selection**:

```python
router_logits, expert_index = torch.max(router_probs, dim=-1, keepdim=True)
```

Note: `router_logits` ở đây bị **overwrite** thành max value. Variable naming tệ; thực ra là routing weight của top-1 expert.

**Step 4: One-hot mask** từ index:

```python
expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)
```

Shape `(B, S, 1, E)`. `1` ở vị trí expert được chọn.

**Step 5: Capacity priority**:

```python
token_priority = torch.cumsum(expert_index, dim=-2)
```

Cumsum dọc seq dim. Cho token thứ `t` đến expert `e`, `token_priority[t][e] = số token từ 0..t đã chọn expert e`. Token sớm có priority thấp (1, 2, 3...).

**Step 6: Mask vượt capacity**:

```python
expert_capacity_mask = token_priority <= self.expert_capacity
expert_index = expert_index * expert_capacity_mask
```

Token có `priority > expert_capacity` bị zero out. Trong dispatch sau, expert_index = 0 nghĩa là không gửi expert nào → drop.

## `SwitchTransformersExperts`

```python
class SwitchTransformersExperts(nn.ModuleDict):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.num_experts = config.num_experts
        for idx in range(config.num_experts):
            self[f"expert_{idx}"] = SwitchTransformersDenseActDense(config)

    def forward(self, hidden_states, selected_experts, routing_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        expert_mask = selected_experts.permute(2, 1, 0)

        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            current_hidden_states = self[f"expert_{expert_idx}"](current_state)
            current_hidden_states = current_hidden_states * routing_weights[token_idx]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```

(class `SwitchTransformersExperts`.)

Khác Mixtral:

1. **ModuleDict thay vì 3D tensor**. `self["expert_0"]`, `self["expert_1"]`, ... mỗi cái là `SwitchTransformersDenseActDense` (T5 FFN).
2. **No `@use_experts_implementation` decorator**. Code thuần eager.
3. **No SwiGLU**. T5 dùng standard FFN (linear + ReLU/GELU + linear).

Vì sao ModuleDict? Lý do legacy: Switch code base từ 2021-2022, trước khi `@use_experts_implementation` xuất hiện. Refactor sang 3D tensor cần effort. Hiện tại HF giữ legacy.

**Lưu ý**: file thực tế là **generated** từ `modular_switch_transformers.py`. Modular file ngắn hơn, kế thừa Mixtral. Header file chính:

```python
#                🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨
#           This file was automatically generated from src/transformers/models/switch_transformers/modular_switch_transformers.py.
#               Do NOT edit this file manually as any edits will be overwritten by the generation of
#             the file from the modular. If any change should be done, please apply the change to the
#                          modular_switch_transformers.py file directly.
```

Nếu fix bug, sửa modular file.

## `SwitchTransformersSparseMLP`

```python
class SwitchTransformersSparseMLP(nn.Module):
    def __init__(self, config: SwitchTransformersConfig):
        super().__init__()
        self.router = SwitchTransformersTop1Router(config)
        self.experts = SwitchTransformersExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        _, selected_experts, routing_weights = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, selected_experts, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
```

Pattern wrap router + experts. Tương tự Mixtral.

## `SwitchTransformersLayerFF`

```python
class SwitchTransformersLayerFF(nn.Module):
    """Switch Transformers Feed Forward layer module.

    Parameters:
        config: SwitchTransformersConfig.
        is_sparse (bool): nếu True dùng sparse MoE, False dùng dense FFN.
    """

    def __init__(self, config, is_sparse=False):
        super().__init__()
        self.is_sparse = is_sparse
        if is_sparse:
            self.mlp = SwitchTransformersSparseMLP(config)
        else:
            self.mlp = SwitchTransformersDenseActDense(config)
        self.layer_norm = SwitchTransformersLayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.mlp(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states
```

(class `SwitchTransformersLayerFF`.)

**`is_sparse` flag**: Switch alternate dense + sparse layer theo pattern. Config có `encoder_sparse_step`, `decoder_sparse_step` chỉ định layer nào sparse.

Ví dụ với `encoder_sparse_step=2`: layer 0 sparse, layer 1 dense, layer 2 sparse, ... Tiết kiệm parameters trên dense layer, giữ sparse cho specialization.

Đây là design choice riêng Switch. Mixtral và hầu hết LLM MoE: mọi layer sparse.

## Encoder vs Decoder

Switch có:

- **Encoder**: layer = self-attention + (sparse_ffn hoặc dense_ffn).
- **Decoder**: layer = self-attention (causal) + cross-attention (to encoder) + (sparse_ffn hoặc dense_ffn).

Cross-attention không phải MoE. Chỉ FFN block là sparse.

## Z-loss

Switch dùng cả aux loss và z-loss:

```python
def router_z_loss_func(router_logits):
    """Compute router z-loss (penalty on large logits magnitude)."""
    num_groups, _ = router_logits.shape
    log_z = torch.logsumexp(router_logits, dim=-1)
    z_loss = log_z ** 2
    return torch.sum(z_loss) / num_groups
```

(class function in `modeling_switch_transformers.py`.)

Coef 0.001. Sum vào total loss:

```python
total_loss = ce_loss + aux_loss_coef * aux_loss + z_loss_coef * z_loss
```

Đã giải thích z-loss ở Phần 1 Chương 4.

## Đặc thù khác

**1. Init weight đặc biệt**:

```python
def _init_weights(self, module):
    if isinstance(module, SwitchTransformersLayerNorm):
        module.weight.data.fill_(factor * 1.0)
    elif isinstance(module, SwitchTransformersDenseActDense):
        d_ff = self.config.d_ff
        d_model = self.config.d_model
        module.wi.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))
        if hasattr(module.wi, "bias") and module.wi.bias is not None:
            module.wi.bias.data.zero_()
        module.wo.weight.data.normal_(mean=0.0, std=factor * ((d_ff) ** -0.5))
        ...
    elif isinstance(module, SwitchTransformersTop1Router):
        module.classifier.weight.data.normal_(mean=0.0, std=factor * 1)
```

(`class SwitchTransformersPreTrainedModel._init_weights`.)

T5-style init: scale theo `d_model^(-0.5)`. Khác Llama (std=0.02 fixed).

**2. Capacity factor configurable**: `expert_capacity` trong config, không phải capacity_factor. User set absolute number (64, 128, ...).

**3. Token dropping at inference**: capacity vẫn được apply ở eval (config `moe_eval_capacity_token_fraction` cho phép tăng/giảm). Switch user chấp nhận quality drop.

## Pitfall

**1. Confuse Switch với Mixtral**: cả hai gọi là "MoE" nhưng paradigm khác. Switch top-1 + capacity; Mixtral top-2 dropless. Don't transfer fine-tune practice directly.

**2. ModuleDict scaling**: với 2048 expert, ModuleDict register 2048 nn.Module. State_dict có 2048 * 2 = 4096 weight names. Loading slow.

**3. Capacity quá nhỏ**: với batch nhỏ, expert_capacity 64 có thể vẫn drop nhiều token. Tăng capacity_factor hoặc giảm num_experts.

**4. Encoder-decoder generate**: Switch dùng `generate()` với `decoder_start_token_id`. Khác decoder-only Mixtral.

**5. Sparse step**: nếu `encoder_sparse_step=4`, chỉ 1/4 encoder layer là sparse. Khi tính active params, tính đúng tỉ lệ.

Chương sau ta đọc DeepSeek-V3.
