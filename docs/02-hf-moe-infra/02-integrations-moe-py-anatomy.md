---
title: integrations/moe.py anatomy
---

# `integrations/moe.py` anatomy

Chương này đi tuần tự `src/transformers/integrations/moe.py`, dừng ở những đoạn quan trọng. Mục tiêu: bạn quen layout, biết tìm đâu khi debug.

## Header

```python
# Copyright 2025 The HuggingFace Team. All rights reserved.
from __future__ import annotations

from collections.abc import Callable
from functools import wraps

from ..utils import logging
from ..utils.generic import GeneralInterface
from ..utils.import_utils import (
    is_torch_available, is_torch_greater_or_equal, is_torch_less_or_equal,
    is_torchdynamo_compiling,
)
from .sonicmoe import sonicmoe_experts_forward

if is_torch_available():
    import torch
    is_torch_greater_or_equal = torch._dynamo.assume_constant_result(is_torch_greater_or_equal)
    is_torch_less_or_equal = torch._dynamo.assume_constant_result(is_torch_less_or_equal)
```

Hai chi tiết:

1. `GeneralInterface` từ `utils.generic` là base class cho mọi interface registry. `AttentionInterface` cũng kế thừa class này.

2. `is_torch_greater_or_equal = torch._dynamo.assume_constant_result(...)`: patch version-check helpers. Dynamo (`torch.compile`) refuse trace `importlib.util.find_spec`. `assume_constant_result` báo Dynamo: function này trả về constant tại compile time, không cần trace body. Khi không patch, compile fail.

## Reference example commented

Lines 43-80 là **comment** chứa code "reference implementation" của một Experts class:

```python
# class Experts(torch.nn.Module):
#     def __init__(self, config):
#         super().__init__()
#         self.num_experts = config.n_routed_experts
#         ...
#     def forward(self, hidden_states, top_k_index, top_k_weights):
#         final_hidden_states = torch.zeros_like(hidden_states)
#         with torch.no_grad():
#             expert_mask = ...
#             expert_hit = ...
#         for expert_idx in expert_hit:
#             ...
#         return final_hidden_states
```

Đây là **canonical eager implementation**. Đọc đây trước khi đọc `batched_mm` / `grouped_mm` để hiểu pattern.

Sao không là code thật? Vì mỗi model có config riêng (`num_local_experts` của Mixtral khác `n_routed_experts` của DeepSeek), reference comment để placeholder. Model thật define class riêng với decorator.

## `_batched_linear`

```python
def _batched_linear(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    is_transposed: bool = False,
) -> torch.Tensor:
    """Batched linear layer supporting optional bias and transposed weights."""
    if is_transposed:
        out = torch.einsum("ebd,end->ebn", input, weight)
    else:
        out = torch.einsum("ebd,ebnd->ebn", input.unsqueeze(0).expand_as(weight[:, None]), weight)
    if bias is not None:
        out = out + bias.unsqueeze(1)
    return out
```

(Lược trích, code thật có comment dài.)

Input shape:

- `input`: `(E, B_e, D)` - mỗi expert nhận `B_e` token, mỗi token `D` dim.
- `weight`: `(E, D, D')` (`is_transposed=True`) hoặc `(E, D', D)` (`is_transposed=False`).
- `bias`: `(E, D')` optional.

Output: `(E, B_e, D')`. Mỗi expert apply linear riêng.

Đây là building block cho `batched_mm_experts_forward`.

## `batched_mm_experts_forward`

```python
def batched_mm_experts_forward(
    self: torch.nn.Module,
    hidden_states: torch.Tensor,
    top_k_index: torch.Tensor,
    top_k_weights: torch.Tensor,
) -> torch.Tensor:
    expert_ids = top_k_index.flatten()  # (S * top_k,)
    ...
    # Clamp EP sentinels so gate_up_proj[expert_ids] stays in-bounds.
    expert_ids.clamp_(0, self.num_experts - 1)

    # Select expert weights
    if self.has_gate:
        selected_weights_gate_up = self.gate_up_proj[expert_ids]
        ...
    else:
        selected_weights_up = self.up_proj[expert_ids]
        ...

    # Token tile: repeat each token top_k times
    selected_hidden_states = hidden_states.repeat_interleave(num_top_k, dim=0)

    # Up projection (batched)
    proj_out = _batched_linear(selected_hidden_states.unsqueeze(0), selected_weights, ...)
    ...

    # Apply gating or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # Down projection
    proj_out = _batched_linear(proj_out, self.down_proj[expert_ids], ...)

    # Weight by routing
    weighted = proj_out.squeeze(0) * sample_weights.unsqueeze(-1)

    # Sum over top_k per token
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)
```

(Lược trích từ `src/transformers/integrations/moe.py`.)

Đặc điểm:

1. **No for-loop**. Mỗi token tile up theo top_k, dispatch batched.
2. **Weight gather by index**. `self.gate_up_proj[expert_ids]` cho ra weight cho mỗi expert mỗi token.
3. **Sentinel handling**. `clamp_(0, num_experts - 1)`: cho EP, expert_ids có thể > num_experts của GPU hiện tại (sentinel). Clamp để indexing không out-of-bounds; routing weight đã zero ở sentinel slot nên output sentinel rows sẽ bị multiply 0.

Đánh đổi với eager loop:

- **Lợi**: vectorized, GPU-friendly. No CPU overhead.
- **Hại**: memory `(S * top_k, hidden)` lớn hơn eager (S, hidden). Mỗi token replicate top_k lần.

Khi batch nhỏ (single decode token), `S = 1`, `top_k = 8` -> 8 rows. Insignificant. Khi batch lớn (prefill 4096 token × top_k 8 = 32k rows), memory cost vẫn OK so với compute.

## `_grouped_mm` ecosystem

Code line 183-337 là implementation `grouped_mm` với fallback:

```python
def _grouped_mm_fallback(input, weight, offs):
    """Fallback grouped matmul khi torch._grouped_mm không có sẵn."""
    output = torch.zeros(input.size(0), weight.size(2), device=input.device, dtype=input.dtype)
    start = 0
    for i, end in enumerate(offs):
        if end > start:
            output[start:end] = input[start:end] @ weight[i]
        start = end
    return output
```

`offs` là cumulative offset: nếu offs = [10, 25, 40], thì rows 0-9 dùng weight[0], 10-24 dùng weight[1], 25-39 dùng weight[2].

```python
if is_torch_available():
    torch.library.custom_op(
        "transformers::grouped_mm_fallback",
        _grouped_mm_fallback,
        mutates_args=(),
        schema="(Tensor input, Tensor weight, Tensor offs) -> Tensor",
    )
    torch.library.register_fake("transformers::grouped_mm_fallback", _grouped_mm_fallback_fake)
    torch.library.register_autograd(
        "transformers::grouped_mm_fallback",
        _grouped_mm_fallback_backward,
        setup_context=_grouped_mm_fallback_setup_context,
    )
```

Register custom op với PyTorch library system để:

1. **Opaque cho torch.compile**. Fallback có for-loop, Dynamo không trace tốt. Wrap thành custom op làm Dynamo treat như black box.
2. **Có autograd**. Tự define backward thay vì để Dynamo derive.
3. **Schema cố định**. Inductor và export biết shape input/output.

`_can_use_grouped_mm`:

```python
def _can_use_grouped_mm(input, weight, offs) -> bool:
    if (is_torchdynamo_compiling() and weight.dtype != torch.bfloat16) or (
        weight.device.type == "cpu" and ...
    ):
        return False
    if weight.device.type == "cuda":
        if hasattr(torch.nn.functional, "grouped_mm"):
            return torch.cuda.get_device_capability(weight.device) >= (8, 0)
        ...
    return hasattr(torch.nn.functional, "grouped_mm") or hasattr(torch, "_grouped_mm")
```

Điều kiện dùng native:

1. PyTorch có `torch.nn.functional.grouped_mm` (2.10+) hoặc `torch._grouped_mm` (2.9+).
2. GPU SM80+ (Ampere trở lên).
3. Compile mode: chỉ bfloat16.
4. CPU: PyTorch 2.11+ (đã fix alignment requirement).

Nếu fail bất cứ điều kiện nào, fallback. Result: code chạy được mọi setup, nhanh khi setup hiện đại.

## `grouped_mm_experts_forward`

Đây là backend chính cho model 2024-2025. Logic phức tạp hơn `batched_mm`:

```python
def grouped_mm_experts_forward(
    self, hidden_states, top_k_index, top_k_weights,
):
    # Sort tokens by expert id (so tokens for same expert are contiguous)
    expert_ids = top_k_index.flatten()
    perm = expert_ids.argsort()
    selected_hidden_states_g = hidden_states[perm // num_top_k]
    sample_weights_g = sample_weights[perm]

    # Compute offsets per expert
    tokens_per_expert = ...
    offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)

    # Sentinel handling
    sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
    expert_ids_g.clamp_(max=self.num_experts - 1)

    # Up projection (grouped)
    proj_out = _grouped_linear(
        selected_hidden_states_g,
        self.gate_up_proj,
        offsets,
        bias=...,
        is_transposed=self.is_transposed,
    )

    # Gate or activation
    if self.has_gate:
        proj_out = self._apply_gate(proj_out)
    else:
        proj_out = self.act_fn(proj_out)

    # Down projection (grouped)
    proj_out = _grouped_linear(proj_out, self.down_proj, offsets, ...)

    # Apply routing weight
    weighted = proj_out * sample_weights_g.unsqueeze(-1)
    weighted.masked_fill_(sentinel_mask, 0.0)

    # Inverse permutation
    inv_perm = torch.empty_like(perm)
    inv_perm[perm] = torch.arange(perm.size(0), device=device)
    weighted = weighted[inv_perm]

    # Reshape + sum: more stable than index_add_ for low precision
    final = weighted.view(num_tokens, num_top_k, hidden_dim).sum(dim=1)
    return final.to(hidden_states.dtype)
```

Khác biệt:

1. **Sort tokens by expert**. `argsort(expert_ids)` group token cùng expert liền nhau. Pre-condition cho `grouped_mm` (kernel expect contiguous groups).
2. **Offset array** thay vì index. Kernel xử lý qua offset, không qua per-row lookup.
3. **Final reduce dùng `reshape+sum`** thay vì `index_add_`. Stable cho fp16/bf16 (index_add atomic không deterministic).

Cost extra: sort O(N log N). Profit: grouped matmul faster than batched cho large N.

## `ExpertsInterface` và singleton

Cuối file:

```python
class ExpertsInterface(GeneralInterface):
    """Interface for registering custom experts forward functions."""

    _global_mapping = {
        "batched_mm": batched_mm_experts_forward,
        "grouped_mm": grouped_mm_experts_forward,
        "sonicmoe": sonicmoe_experts_forward,
    }

    def get_interface(self, experts_implementation: str, default: Callable) -> Callable:
        if experts_implementation is None:
            logger.warning_once(...)
        elif experts_implementation != "eager" and experts_implementation not in self:
            raise KeyError(...)
        return super().get(experts_implementation, default)


ALL_EXPERTS_FUNCTIONS = ExpertsInterface()
```

`sonicmoe`: backend thử nghiệm dùng custom CUDA kernel của Mosaic/IBM (chuỗi này không đi sâu).

`ALL_EXPERTS_FUNCTIONS` là singleton. User dispatch qua `config._experts_implementation = "grouped_mm"`.

## `use_experts_implementation` decorator

```python
def use_experts_implementation(
    experts_class=None,
    *,
    experts_interface=ALL_EXPERTS_FUNCTIONS,
    is_concatenated=True,
    is_transposed=False,
    has_bias=False,
    has_gate=True,
):
    def wrapper(experts_class):
        original_init = experts_class.__init__
        original_forward = experts_class.forward

        @wraps(original_init)
        def __init__(self, config, *args, **kwargs):
            original_init(self, config, *args, **kwargs)
            self.config = config
            self.has_gate = has_gate
            self.has_bias = has_bias
            self.is_transposed = is_transposed
            self.is_concatenated = is_concatenated

        @wraps(original_forward)
        def forward(self, *args, **kwargs):
            experts_forward = experts_interface.get_interface(
                self.config._experts_implementation, original_forward
            )
            return experts_forward(self, *args, **kwargs)

        if not hasattr(experts_class, "_apply_gate"):
            experts_class._apply_gate = _default_apply_gate

        experts_class.__init__ = __init__
        experts_class.forward = forward
        return experts_class

    if experts_class is not None:
        return wrapper(experts_class)
    return wrapper
```

Decorator nhận class, return class đã modify:

1. **`__init__`** gói lại `original_init` rồi attach flags.
2. **`forward`** gói lại `original_forward`, nhưng khi gọi sẽ dispatch qua interface.
3. **`_apply_gate`** mặc định là `_default_apply_gate` (split chunk, act gate, mul up).

Hai cách dùng:

```python
@use_experts_implementation
class MixtralExperts(nn.Module): ...

@use_experts_implementation(is_concatenated=False, has_bias=True)
class GptOssExperts(nn.Module): ...
```

Cách 1 dùng default flags. Cách 2 specify.

## Tổng kết chương

File `integrations/moe.py` có ba mảng:

1. **Hai backend forward** (`batched_mm`, `grouped_mm`) cho expert dispatch.
2. **`ExpertsInterface`** singleton để swap backend qua config.
3. **Decorator** rewrite expert class hoá dispatch transparent.

Mỗi model MoE 2024+ chỉ cần:

1. Define expert class với weight 3D tensor.
2. Apply decorator với flags đúng.
3. Define eager forward (cho fallback).

Phần còn lại do infrastructure handle.

Chương sau ta đi sâu vào `ExpertsInterface`.
