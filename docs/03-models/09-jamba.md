---
title: Jamba
---

# Jamba

Jamba (AI21 Labs, March 2024) là model hybrid đầu tiên kết hợp **Mamba (SSM) + Transformer + MoE**. Mỗi block có thể là Mamba layer, attention layer, hoặc MoE FFN layer. Đại diện cho hướng kiến trúc composition.

## Context

- **Tác giả**: AI21 Labs.
- **Release**: March 2024.
- **Paper**: "Jamba: A Hybrid Transformer-Mamba Language Model".
- **License**: Apache 2.0.
- **Variants**: Jamba-v0.1 (52B total, 12B active), Jamba-1.5-Mini, Jamba-1.5-Large.

Highlight: 256k context window stable. State Space Model (Mamba) handle long context efficiently.

## Config key

```python
class JambaConfig:
    hidden_size = 4096
    intermediate_size = 14336
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8           # GQA
    num_experts = 16
    num_experts_per_tok = 2
    expert_layer_period = 2            # MoE mỗi 2 layer
    expert_layer_offset = 1            # Bắt đầu từ layer 1
    attn_layer_period = 8              # Attention mỗi 8 layer
    attn_layer_offset = 4              # Bắt đầu từ layer 4
    # Pattern resulting:
    # layer 0: Mamba + MLP
    # layer 1: Mamba + MoE
    # layer 2: Mamba + MLP
    # layer 3: Mamba + MoE
    # layer 4: Attention + MLP
    # layer 5: Attention + MoE
    # ...
    mamba_d_state = 16
    mamba_d_conv = 4
    mamba_expand = 2
    vocab_size = 65536
```

## Cấu trúc

```
modeling_jamba.py (952 dòng)
├── JambaRMSNorm
├── JambaMambaMixer                 # SSM core
├── JambaAttention                  # GQA
├── JambaMLP                        # Dense FFN
├── JambaSparseMoeBlock             # Standard MoE
├── JambaAttentionDecoderLayer      # Attention + (MLP hoặc MoE)
├── JambaMambaDecoderLayer          # Mamba + (MLP hoặc MoE)
├── JambaPreTrainedModel
├── JambaModel
└── JambaForCausalLM
```

## Layer pattern

Jamba có **hai trục alternation**:

1. **Attention vs Mamba** (across layers).
2. **MoE vs dense MLP** (across layers).

Mỗi layer có exact 1 trong 4 combinations:

- `Mamba + MLP`
- `Mamba + MoE`
- `Attention + MLP`
- `Attention + MoE`

Pattern config bởi `expert_layer_period`, `attn_layer_period`, offsets.

## `JambaMambaMixer` (SSM)

```python
class JambaMambaMixer(nn.Module):
    """Mamba State Space Model mixer."""

    def __init__(self, config, layer_idx):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.mamba_expand * config.hidden_size  # 2x
        self.d_state = config.mamba_d_state    # 16
        self.d_conv = config.mamba_d_conv       # 4

        # In projection: hidden -> 2 * intermediate (gate + x)
        self.in_proj = nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)

        # 1D conv
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            kernel_size=self.d_conv,
            groups=self.intermediate_size,
            padding=self.d_conv - 1,
        )

        # State Space parameters
        self.x_proj = nn.Linear(self.intermediate_size, self.d_state * 2 + 1)  # B, C, delta
        self.dt_proj = nn.Linear(1, self.intermediate_size)
        self.A_log = nn.Parameter(...)
        self.D = nn.Parameter(...)

        # Out projection
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
```

(Lược trích pattern từ `src/transformers/models/jamba/modeling_jamba.py`.)

Mamba forward không phải attention. Recurrent state update:

```
h_t = A * h_{t-1} + B * x_t
y_t = C * h_t + D * x_t
```

A, B, C, D learnable. h_t là hidden state (size d_state, default 16).

Long context efficient vì O(N) (recurrent), không phải O(N²) attention.

## `JambaSparseMoeBlock`

```python
class JambaSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_experts
        self.top_k = config.num_experts_per_tok
        self.router = nn.Linear(self.hidden_dim, self.num_experts, bias=False)
        self.experts = nn.ModuleList([JambaMLP(config) for _ in range(self.num_experts)])

    def forward(self, hidden_states):
        ...
```

(`class JambaSparseMoeBlock`.)

Khác Mixtral và post-2024 model: dùng `nn.ModuleList[JambaMLP]` thay vì 3D weight tensor. Tương tự Switch (legacy).

Lý do: Jamba release 2024 sớm, trước HF refactor toàn diện. Khi infrastructure mới ra, Jamba chưa migrate.

## `JambaAttentionDecoderLayer`

```python
class JambaAttentionDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        num_experts_per_tok = config.num_experts_per_tok if (config.num_experts > 1) else 1
        ffn_layer_class = JambaSparseMoeBlock if (
            (layer_idx - config.expert_layer_offset) % config.expert_layer_period == 0
        ) else JambaMLP

        self.self_attn = JambaAttention(config, layer_idx)
        self.feed_forward = ffn_layer_class(config)
        ...

    def forward(self, hidden_states, ...):
        ...
        attn_output = self.self_attn(...)
        ...
        ffn_output = self.feed_forward(...)
        ...
```

Layer này dùng attention. FFN có thể dense hoặc MoE tuỳ index.

## `JambaMambaDecoderLayer`

```python
class JambaMambaDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.mamba = JambaMambaMixer(config, layer_idx)
        ffn_layer_class = JambaSparseMoeBlock if (
            (layer_idx - config.expert_layer_offset) % config.expert_layer_period == 0
        ) else JambaMLP
        self.feed_forward = ffn_layer_class(config)
        ...
```

Layer này dùng Mamba. FFN cùng pattern (dense hoặc MoE).

## `JambaModel`

```python
class JambaModel(JambaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        ...
        layers = []
        for i in range(config.num_hidden_layers):
            if (i - config.attn_layer_offset) % config.attn_layer_period == 0:
                layers.append(JambaAttentionDecoderLayer(config, layer_idx=i))
            else:
                layers.append(JambaMambaDecoderLayer(config, layer_idx=i))
        self.layers = nn.ModuleList(layers)
```

(`class JambaModel.__init__`.)

Construct layer list theo pattern. Default `attn_layer_period=8`, `attn_layer_offset=4`: layer 4, 12, 20, 28 là attention; còn lại Mamba.

Total ~28 Mamba layers + 4 attention layers + 16 MoE FFN layers + 16 dense MLP layers (32 layer total, 50% MoE).

## Cache management (rất khác)

Jamba có **hai loại cache**:

1. **Attention KV cache**: như Llama/Mixtral standard.
2. **Mamba SSM cache**: state `h_t` của recurrent, shape `(B, intermediate, d_state)`.

`HybridMambaAttentionDynamicCache` class wrap cả hai. Generate có logic phân biệt layer attention vs Mamba.

```python
class HybridMambaAttentionDynamicCache:
    def __init__(self, config, ...):
        self.key_cache = []
        self.value_cache = []
        self.conv_states = []     # Mamba conv state
        self.ssm_states = []      # Mamba SSM state
        ...
```

(Pseudocode pattern.)

Phức tạp hơn DynamicCache standard. Generate code Jamba có dispatch riêng.

## `JambaForCausalLM`

```python
class JambaForCausalLM(JambaPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.model = JambaModel(config)
        self.lm_head = nn.Linear(...)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        ...

    def forward(self, ..., output_router_logits=None, ...):
        outputs = self.model(...)
        ...
        if labels is not None:
            loss = self.loss_function(logits, labels, ...)
            if output_router_logits:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits,
                    self.config.num_experts,
                    self.config.num_experts_per_tok,
                    ...
                )
                loss += self.router_aux_loss_coef * aux_loss
```

Standard aux loss. Coef 0.001.

## Vì sao hybrid

Mamba có:

- **Pro**: O(N) compute, long context, no KV cache (chỉ SSM state).
- **Con**: weaker in-context learning (so với attention), khó debug.

Attention có:

- **Pro**: strong few-shot, in-context learning.
- **Con**: O(N²) compute, large KV cache.

Hybrid: 7-8 Mamba layer mỗi 1 attention layer. Most compute trong Mamba (long context cheap), occasional attention (capability boost).

MoE thêm vào để tăng capacity mà active params giữ nhỏ. Mỗi 2 layer có MoE.

Result: model 52B total params, 12B active. Context 256k stable. Quality competitive với dense 30B.

## Pitfall

**1. Cache complexity**: nếu fine-tune Jamba với Trainer, cache management custom. Default Trainer code có thể không handle.

**2. Layer pattern hard-code**: `attn_layer_period=8`, `expert_layer_period=2`. Nếu fork với pattern khác, phải config carefully.

**3. Mamba kernel**: cần `mamba-ssm` package hoặc fall back PyTorch slow. Check install.

**4. MoE chỉ ở layer chẵn (offset 1)**: 16 layer MoE, không 32. Active params calc phải tính đúng.

**5. Generate cache lifetime**: attention cache grow theo seq, Mamba state cố định. Different memory profile.

**6. Pretrain recipe phức tạp**: aux loss với 50% layer là MoE, balance khó. AI21 paper khuyến cao coef 0.001 stable.

Chương sau ta đọc NLLB-MoE.
