---
title: GPT-OSS
---

# GPT-OSS

GPT-OSS (OpenAI, August 2025) là dòng MoE open-weight đầu tiên của OpenAI. Hai variant: GPT-OSS-20B (3.6B active) và GPT-OSS-120B (5.1B active). Đặc thù: weight format `is_transposed`, có bias, custom gate với clamp, và MXFP4 quantization native. Đại diện cho design "production-aware quantization".

## Context

- **Tác giả**: OpenAI.
- **Release**: August 2025.
- **Paper**: "GPT-OSS model card" (OpenAI, 2025).
- **License**: Apache 2.0.
- **Variants**: GPT-OSS-20B (20B total), GPT-OSS-120B (117B total).

## Config key

GPT-OSS-120B:

```python
class GptOssConfig:
    hidden_size = 2880
    intermediate_size = 2880               # mỗi expert FFN size = hidden (small!)
    num_hidden_layers = 36
    num_attention_heads = 64
    num_key_value_heads = 8                # GQA 8:1
    num_local_experts = 128
    num_experts_per_tok = 4                # top-4
    sliding_window = 128
    rms_norm_eps = 1e-5
    rope_scaling = None
    head_dim = 64
    attention_bias = True                  # có bias (OpenAI style)
    quantization_config = Mxfp4Config(...)
    vocab_size = 201088
```

GPT-OSS-20B:

```python
hidden_size = 2880
num_hidden_layers = 24
num_local_experts = 32
num_experts_per_tok = 4
```

## Cấu trúc

```
modeling_gpt_oss.py (712 dòng)
├── GptOssRMSNorm
├── GptOssExperts                  # 3D weight, has_bias, is_transposed, @use_experts_implementation
├── GptOssTopKRouter               # Linear with bias + sigmoid (sort of)
├── GptOssMLP                       # Wrap router + experts
├── GptOssAttention                # GQA with sliding window
├── GptOssRotaryEmbedding
├── GptOssDecoderLayer
├── GptOssPreTrainedModel
├── GptOssModel
└── GptOssForCausalLM
```

## `GptOssExperts` (đáng chú ý)

```python
@use_experts_implementation(is_concatenated=False, is_transposed=True, has_bias=True)
class GptOssExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.hidden_size = config.hidden_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_size, 2 * self.intermediate_size))
        self.gate_up_proj_bias = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_size))
        self.down_proj = nn.Parameter(torch.empty((self.num_experts, self.intermediate_size, self.hidden_size)))
        self.down_proj_bias = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))
        self.alpha = 1.702
        self.limit = 7.0
```

(`src/transformers/models/gpt_oss/modeling_gpt_oss.py`, class `GptOssExperts`.)

**Decorator flags**:

- `is_concatenated=False`: gate_up vẫn là single tensor nhưng layout interleave (gate0, up0, gate1, up1, ...).
- `is_transposed=True`: shape `(E, hidden, 2*intermediate)` thay vì `(E, 2*intermediate, hidden)`.
- `has_bias=True`: có bias cho gate_up và down.

Weight shape:

- `gate_up_proj`: `(128, 2880, 5760)`. Transposed.
- `gate_up_proj_bias`: `(128, 5760)`.
- `down_proj`: `(128, 2880, 2880)`. Transposed.
- `down_proj_bias`: `(128, 2880)`.

**Bias for production**: bias giúp với 4-bit quantization. Khi quantize weight về MXFP4, bias remains higher precision, compensate quant error.

### `_apply_gate` custom

```python
def _apply_gate(self, gate_up: torch.Tensor) -> torch.Tensor:
    gate, up = gate_up[..., ::2], gate_up[..., 1::2]
    gate = gate.clamp(min=None, max=self.limit)
    up = up.clamp(min=-self.limit, max=self.limit)
    glu = gate * torch.sigmoid(gate * self.alpha)
    gated_output = (up + 1) * glu
    return gated_output
```

(method `GptOssExperts._apply_gate`.)

Đặc thù:

**1. Interleaved split**: `gate_up[..., ::2]` lấy chỉ số chẵn, `gate_up[..., 1::2]` lấy chỉ số lẻ. Khác `chunk(2, dim=-1)` (split nửa đầu / nửa sau).

Đây là pattern "interleaved" tương ứng `is_concatenated=False` trong decorator.

**2. Clamp**:

```python
gate = gate.clamp(min=None, max=self.limit)         # gate <= 7
up = up.clamp(min=-self.limit, max=self.limit)      # -7 <= up <= 7
```

Tránh activation explode. Hữu ích cho 4-bit quantization: extreme values bị clip, range nhỏ → quantize chính xác hơn.

**3. SwiGLU variant**:

```python
glu = gate * torch.sigmoid(gate * self.alpha)
gated_output = (up + 1) * glu
```

So với SwiGLU standard: `act(gate) * up` với `act = silu = x * sigmoid(x)`.

GPT-OSS: `gate * sigmoid(gate * alpha)` với `alpha = 1.702`. Đây là **GELU approximation** (`gelu(x) ≈ x * sigmoid(1.702 * x)`). OpenAI dùng GELU thay SwiGLU silu.

Và `(up + 1) * glu` thay vì `up * glu`. `+1` là offset trick: nếu `up = 0`, output = `glu` (không zero); như identity gate. Smooth fallback.

Đây là kết quả của empirical tuning. Khác model khác.

### Forward (eager)

```python
def forward(self, hidden_states, router_indices=None, routing_weights=None):
    next_states = torch.zeros_like(hidden_states)
    with torch.no_grad():
        expert_mask = torch.nn.functional.one_hot(router_indices, num_classes=self.num_experts)
        expert_mask = expert_mask.permute(2, 1, 0)
        expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

    for expert_idx in expert_hit:
        expert_idx = expert_idx[0]
        if expert_idx == self.num_experts:
            continue
        top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
        current_state = hidden_states[token_idx]
        # Note: matmul direct (transposed format), không F.linear
        gate_up = current_state @ self.gate_up_proj[expert_idx] + self.gate_up_proj_bias[expert_idx]
        gated_output = self._apply_gate(gate_up)
        out = gated_output @ self.down_proj[expert_idx] + self.down_proj_bias[expert_idx]
        weighted_output = out * routing_weights[token_idx, top_k_pos, None]
        next_states.index_add_(0, token_idx, weighted_output.to(hidden_states.dtype))

    return next_states
```

(method `GptOssExperts.forward`.)

Khác Mixtral:

1. `current_state @ self.gate_up_proj[expert_idx]` thay vì `F.linear(...)`. Vì transposed format, direct matmul đúng dimension.
2. Cộng bias explicit.
3. Gọi `self._apply_gate(gate_up)` (custom).

## `GptOssTopKRouter`

```python
class GptOssTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))
        self.bias = nn.Parameter(torch.zeros(self.num_experts))

    def forward(self, hidden_states):
        router_logits = F.linear(hidden_states, self.weight, self.bias)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_scores = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_top_value.dtype)
        return router_logits, router_scores, router_indices
```

(class `GptOssTopKRouter`.)

Đặc thù:

**1. Router có bias**: `nn.Linear(hidden, num_experts, bias=True)`. Bias giúp compensate khi quantize.

**2. Top-k trên logits, không trên probs**:

```python
router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
```

Khác Mixtral (topk sau softmax). Argmax/topk monotonic với softmax → result giống. Khác biệt: GPT-OSS chỉ softmax **trên top-k** scores, không trên toàn distribution.

**3. Softmax local**:

```python
router_scores = torch.nn.functional.softmax(router_top_value, dim=1)
```

Softmax qua `top_k` chiều cuối. Output sum = 1 trong top-k. Tương đương Mixtral renormalize.

Lợi: ít compute (chỉ softmax top-4 thay vì 128). Quality cùng.

## `GptOssMLP` (SparseMoeBlock)

```python
@use_kernel_forward_from_hub("MegaBlocksMoeMLP")
class GptOssMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.router = GptOssTopKRouter(config)
        self.experts = GptOssExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.reshape(-1, hidden_dim)
        _, router_scores, router_indices = self.router(hidden_states)
        hidden_states = self.experts(hidden_states, router_indices, router_scores)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_scores
```

(class `GptOssMLP`.)

**`@use_kernel_forward_from_hub("MegaBlocksMoeMLP")`**: thử dùng custom kernel từ HuggingFace Hub (megablocks kernel) khi available. Fallback về eager nếu không.

Đây là pattern "kernels-on-hub" cho phép distribute kernel optimized qua HF Hub thay vì compile lúc install.

## MXFP4 quantization

GPT-OSS native quant với MXFP4 (Microscaling FP4):

```python
quantization_config = Mxfp4Config(
    modules_to_not_convert=[...],   # router, embed, lm_head not quantized
)
```

Logic ở `src/transformers/integrations/mxfp4.py`. Walkthrough chi tiết ở Phần 4 Chương 4.

Tóm tắt: weight expert (`gate_up_proj`, `down_proj`) được store ở 4-bit MXFP4 (E2M1 với scale block 32). Activation và bias giữ bf16. Inference compute dequant on-the-fly.

Memory cost của weight: 4-bit vs 16-bit = 4x saving. GPT-OSS-120B với MXFP4: ~60GB cho weight (vs ~234GB bf16). Fit trên 2 H100.

## `GptOssDecoderLayer`

```python
class GptOssDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMLP(config)
        ...

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, ... = self.self_attn(hidden_states, ...)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        mlp_output, router_scores = self.mlp(hidden_states)
        hidden_states = residual + mlp_output
        return hidden_states, router_scores, ...
```

Standard pattern. Mọi layer MoE.

## `GptOssAttention`

```python
class GptOssAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_proj = nn.Linear(..., bias=config.attention_bias)
        self.k_proj = nn.Linear(..., bias=config.attention_bias)
        self.v_proj = nn.Linear(..., bias=config.attention_bias)
        self.o_proj = nn.Linear(..., bias=config.attention_bias)
        self.sinks = nn.Parameter(torch.empty(num_heads))   # Attention sinks
        self.sliding_window = config.sliding_window if (layer_idx + 1) % 2 == 0 else None
```

(class `GptOssAttention`.)

Đặc thù:

1. **Attention bias**: `bias=True` cho QKVO. OpenAI style.
2. **Attention sinks**: parameter scalar mỗi head. Tạo "sink" token virtual để attention không degenerate khi context dài. Tham khảo "Attention Sinks" paper (Han et al., 2024).
3. **Alternating sliding window**: layer chẵn dùng full attention, layer lẻ dùng sliding window 128. Hybrid để giảm cost long context.

## Pitfall

**1. Quên `is_transposed=True` trong decorator**: nếu fork và bỏ flag, weight layout không khớp → wrong math silently.

**2. Interleaved split**: `gate_up[..., ::2]` không phải `chunk(2)`. Hai cách split khác nhau hoàn toàn cho cùng tensor.

**3. MXFP4 quantize chỉ expert**: router weight giữ fp16/bf16. Quantize router gây route collapse. Đảm bảo `modules_to_not_convert` include router.

**4. Clamp limit = 7**: nếu fine-tune và tăng learning rate, gate values có thể vượt 7. Output bị clamp, gradient có thể vanish.

**5. Custom `_apply_gate` không tương thích với `grouped_mm`**: nếu config dùng grouped backend, infrastructure call `_apply_gate` ở input format khác. Phải test cả ba mode.

**6. Sliding window mỗi layer khác**: cache shape phải handle full + sliding. Complicate KV cache.

Chương sau ta đọc OLMoE (5 supplementary đầu).
