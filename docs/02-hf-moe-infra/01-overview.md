---
title: Tổng quan Phần 2
---

# Phần 2: HuggingFace MoE infrastructure

Trước khi đọc 10 model trong Phần 3, ta phải hiểu **infrastructure dùng chung**. HuggingFace đã refactor năm 2024-2025 để tách logic dispatch expert ra khỏi từng model, đặt vào `src/transformers/integrations/moe.py`. Hiểu file này, đọc 10 model nhanh hơn nhiều.

## Mục tiêu Phần 2

Sau Phần 2, bạn:

1. Biết `integrations/moe.py` chứa gì, vì sao tồn tại, ai dùng.
2. Hiểu `ExpertsInterface` parallel với `AttentionInterface`.
3. Biết khi nào dispatch dùng `batched_mm` vs `grouped_mm`, vì sao.
4. Hiểu decorator `@use_experts_implementation` rewrite class như thế nào.
5. Biết `load_balancing_loss_func` helper được share giữa các model.

## Cấu trúc Phần 2

- Chương 2: `integrations/moe.py` anatomy. Tour toàn file 583 dòng.
- Chương 3: `ExpertsInterface` và decorator `use_experts_implementation`.
- Chương 4: `batched_mm` vs `grouped_mm` so sánh.
- Chương 5: `load_balancing_loss_func` helper.

## Vì sao infrastructure tách

Trước refactor, mỗi model MoE viết riêng:

- `MixtralExperts.forward`: for-loop qua expert, `index_add_`.
- `Qwen2MoeSparseMoeBlock.forward`: tương tự nhưng class riêng.
- `DeepseekV2MoE.forward`: thêm shared expert nhưng base giống.

Hàng tá model duplicate cùng logic. Mỗi bug fix phải sync nhiều file. Mỗi optimization (như `grouped_mm` mới) phải copy vào mọi model.

Solution: tách dispatch logic vào một file shared. Mỗi model định nghĩa **weight layout** (3D tensor), gọi infrastructure để dispatch.

## Parallel với AttentionInterface

Pattern này giống đúc `AttentionInterface` (Phần 1 chuỗi bài giảng nền):

| | Attention | MoE |
|---|---|---|
| Interface class | `AttentionInterface` | `ExpertsInterface` |
| Global mapping | `ALL_ATTENTION_FUNCTIONS` | `ALL_EXPERTS_FUNCTIONS` |
| Backend list | eager, sdpa, flash_attention_2, flex, paged | eager, batched_mm, grouped_mm |
| Selection knob | `config._attn_implementation` | `config._experts_implementation` |
| Per-class flag | `_supports_sdpa`, ... | (none, runtime check via try/except) |

Cùng triết lý: model định nghĩa data layout, interface dispatch implementation.

## Anatomy nhanh `integrations/moe.py`

```
src/transformers/integrations/moe.py (583 dòng)
├── Imports + torch dynamo patches             (~40 dòng)
├── Reference example commented (Experts class) (~38 dòng)
├── _batched_linear helper                      (~32 dòng)
├── batched_mm_experts_forward                  (~64 dòng) <- backend 1
├── _grouped_mm_fallback + custom op            (~80 dòng)
├── _can_use_grouped_mm                         (~44 dòng)
├── _grouped_mm dispatcher                      (~30 dòng)
├── _grouped_linear helper                      (~36 dòng)
├── grouped_mm_experts_forward                  (~100 dòng) <- backend 2
├── ExpertsInterface class                      (~22 dòng)
├── ALL_EXPERTS_FUNCTIONS singleton             (1 dòng)
├── _default_apply_gate helper                  (~12 dòng)
└── use_experts_implementation decorator        (~60 dòng) <- entrypoint
```

Hai backend chính:

- `batched_mm_experts_forward`: dùng `_batched_linear` (đơn giản, hoạt động mọi setup).
- `grouped_mm_experts_forward`: dùng `torch._grouped_mm` hoặc fallback (hiệu quả với SM80+).

Một entrypoint chính: `use_experts_implementation` decorator.

## Sneak peek: cách model dùng

Từ Mixtral:

```python
from ...integrations import use_experts_implementation

@use_experts_implementation
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(...)
        self.down_proj = nn.Parameter(...)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager implementation: for-loop + index_add_
        ...
```

Decorator `@use_experts_implementation` wrap `__init__` (thêm flag `has_gate`, `has_bias`, ...) và wrap `forward` (lookup interface dispatch).

Khi `config._experts_implementation == "grouped_mm"`, forward bypass eager loop, gọi `grouped_mm_experts_forward(self, ...)`. Khi config "eager", giữ original forward (loop).

## Vì sao nhiều backend?

Mỗi backend phù hợp scenario khác:

- **Eager** (for-loop): debug, batch nhỏ, không cần kernel.
- **batched_mm**: medium batch, không yêu cầu PyTorch mới.
- **grouped_mm**: large batch, PyTorch 2.9+, GPU SM80+.

`config._experts_implementation` chọn explicit hoặc auto-select.

## Một số model "concatenated" vs "interleaved"

Decorator có flag `is_concatenated`:

```python
@use_experts_implementation(is_concatenated=True)
class MixtralExperts(nn.Module): ...

@use_experts_implementation(is_concatenated=False, is_transposed=True, has_bias=True)
class GptOssExperts(nn.Module): ...
```

- `is_concatenated=True`: weight layout `[gate_proj; up_proj]` được concat thành một matrix `(2*d_ff, hidden)`. Single linear xuất `(2*d_ff)` rồi chunk thành (gate, up). Tiết kiệm một call linear.
- `is_concatenated=False`: weight tách `[gate_proj, up_proj]` riêng, hai linear call. Compat cho model legacy.

- `is_transposed=False` (Mixtral): weight shape `(E, out, in)`, dùng `F.linear` convention.
- `is_transposed=True` (GPT-OSS): weight shape `(E, in, out)`, dùng matmul native không transpose.

- `has_bias=False` (default): không bias trong expert linear.
- `has_bias=True` (GPT-OSS): có bias cho gate_up + down.

- `has_gate=False`: expert không có gating (SwiGLU), chỉ `act(up_proj(x))`. Hiếm; default True.

Mỗi model bật đúng flag, code infrastructure tự handle. Đây là điểm gây nhầm lẫn nhất khi đọc.

## Liên kết Phần 3

Khi đọc Phần 3 chương Mixtral, sẽ thấy `@use_experts_implementation` (default flags). Khi đọc GPT-OSS, sẽ thấy `@use_experts_implementation(is_concatenated=False, is_transposed=True, has_bias=True)`. Hiểu Phần 2 trước, đọc Phần 3 không bị bỡ ngỡ.

Chương sau ta tour toàn bộ `integrations/moe.py`.
