---
title: batched_mm vs grouped_mm
---

# `batched_mm` vs `grouped_mm`

Hai backend chính của `ExpertsInterface`. Cả hai đều dispatch expert computation không có for-loop Python, nhưng kỹ thuật khác nhau. Chương này so sánh: algorithm, performance, khi nào dùng cái nào.

## Recap input

Cả hai backend nhận:

```python
hidden_states: (num_tokens, hidden_dim)       # đã flatten batch*seq
top_k_index: (num_tokens, top_k)              # int, expert ID
top_k_weights: (num_tokens, top_k)            # float, routing prob (đã renormalize)
```

Và truy cập (qua `self`):

```python
self.gate_up_proj: (num_experts, 2*intermediate, hidden)
self.down_proj: (num_experts, hidden, intermediate)
self.num_experts: int
```

Trả về:

```python
final_hidden_states: (num_tokens, hidden_dim)
```

Mỗi token đi qua `top_k` expert, output là weighted sum.

## `batched_mm` algorithm

```
1. Flatten top_k_index -> expert_ids (N*k,)
2. Gather weight per token:
   selected_weights = self.gate_up_proj[expert_ids]
   # Shape: (N*k, 2*intermediate, hidden)
3. Tile hidden_states by top_k:
   selected_hidden = hidden_states.repeat_interleave(top_k, dim=0)
   # Shape: (N*k, hidden)
4. Batched linear:
   proj_out = _batched_linear(selected_hidden, selected_weights)
   # Each (token, expert) pair gets its own matmul
   # Shape: (N*k, 2*intermediate)
5. Apply gate, get (N*k, intermediate)
6. Gather down weight:
   selected_down = self.down_proj[expert_ids]
   # Shape: (N*k, hidden, intermediate)
7. Batched linear (down):
   proj_out = _batched_linear(proj_out, selected_down)
   # Shape: (N*k, hidden)
8. Weight + sum per token:
   weighted = proj_out * top_k_weights.flatten().unsqueeze(-1)
   # Shape: (N*k, hidden)
   final = weighted.view(N, top_k, hidden).sum(dim=1)
```

**Cost memory**: gather `selected_weights` size `(N*k, 2*intermediate, hidden) = N*k * 2*d_ff * d`. Với `N=4096, k=8, d_ff=14336, d=4096`: `4096 * 8 * 28672 * 4096 = 3.8 TB`. Không khả thi.

Wait, không. PyTorch fancy indexing `self.gate_up_proj[expert_ids]` không tạo full tensor; nó tạo **view** trỏ tới same memory với mới stride. **No memory blow up**.

Thực ra, fancy indexing có thể copy hoặc view tuỳ shape. Trong trường hợp này, expert_ids là 1D int, nên copy thành new tensor (N*k, 2*intermediate, hidden). Memory blow up thật.

**Cost compute**: `N*k` matmul `(1, hidden) @ (hidden, 2*intermediate)`. Mỗi matmul nhỏ. Tổng FLOPs ~ `N*k * hidden * 2*intermediate = 4096 * 8 * 4096 * 28672 ≈ 3.85 TFLOPs`. So với dense Llama-3 (FFN ratio ~3.5): `N * hidden * 2*intermediate = 4096 * 4096 * 28672 ≈ 0.48 TFLOPs`. Eight times more cho top-k = 8.

Trade-off của `batched_mm`:

- **Lợi**: code đơn giản, không cần sort.
- **Hại**: memory blow up cho gather, nhiều matmul nhỏ (GPU không full utilization).

## `grouped_mm` algorithm

```
1. Flatten expert_ids = top_k_index.flatten() -> (N*k,)
2. Sort: perm = expert_ids.argsort()
   sorted_expert_ids = expert_ids[perm]
   # Sau sort: rows cùng expert nằm liền nhau
3. Reorder hidden:
   sorted_hidden = hidden_states[perm // top_k]
   # Token tile theo perm, nhưng tile theo expert
4. Compute offsets per expert:
   tokens_per_expert = bincount(sorted_expert_ids, minlength=num_experts)
   offsets = cumsum(tokens_per_expert)
   # offsets[i] = số token đến expert 0..i
5. Sentinel mask cho EP:
   sentinel_mask = (sorted_expert_ids >= num_experts).unsqueeze(-1)
   sorted_expert_ids.clamp_(max=num_experts - 1)
6. Up projection grouped:
   proj_out = _grouped_linear(sorted_hidden, self.gate_up_proj, offsets)
   # Kernel: cho mỗi (start, end) trong offsets, matmul vùng [start:end]
   # Shape: (N*k, 2*intermediate)
7. Apply gate -> (N*k, intermediate)
8. Down projection grouped:
   proj_out = _grouped_linear(proj_out, self.down_proj, offsets)
9. Weight: weighted = proj_out * sorted_weights.unsqueeze(-1)
10. Mask sentinel: weighted.masked_fill_(sentinel_mask, 0.0)
11. Inverse perm:
    inv_perm[perm] = arange(N*k)
    unsorted = weighted[inv_perm]
12. Reshape + sum:
    final = unsorted.view(N, top_k, hidden).sum(dim=1)
```

**Cost memory**: không gather weight (`self.gate_up_proj` accessed via offset, no replicate). Sorted hidden_states là (N*k, hidden), one copy. Significant memory saving.

**Cost compute**: cùng FLOPs với batched_mm (vẫn N*k token × matmul), nhưng grouped kernel:

- **Batch matmul lớn** (mỗi expert có hàng trăm-nghìn token). GPU full utilization.
- **No per-row weight load**. Read weight once per expert.

PyTorch `torch._grouped_mm` (2.9+) / `torch.nn.functional.grouped_mm` (2.10+) implement này. Native CUDA kernel ~2-5x faster than naive batched on large batch.

## Sentinel mechanism (EP)

```python
sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
expert_ids_g.clamp_(max=self.num_experts - 1)
```

Khi distributed với EP, mỗi GPU chỉ giữ subset expert. Ví dụ GPU 0 giữ expert 0-31, GPU 1 giữ 32-63. Khi router output `expert_id = 50` cho một token trên GPU 0, đó là "sentinel" (expert không trên GPU này).

Token sentinel sẽ:

1. Có `routing_weight = 0` (router output đã được mask trước).
2. Có `expert_id` ngoài `[0, num_experts)`.

Clamp `expert_id` về `num_experts - 1` để indexing không out-of-bounds. Sau dispatch, `weighted_out * 0 = 0` (vì routing weight = 0), `masked_fill_(sentinel_mask, 0)` reinforce.

Output cho sentinel = 0, đúng (token này sẽ nhận expert output từ GPU khác qua all-to-all). Vai trò của all-to-all communication: aggregate output từ mọi GPU theo token id.

## Reshape+sum vs index_add_

```python
# grouped_mm dùng reshape+sum
final = weighted.view(N, top_k, hidden).sum(dim=1)

# batched_mm cũng dùng reshape+sum sau decorator update
# (legacy eager dùng index_add_)
```

Lý do bỏ `index_add_`:

1. **Non-deterministic CUDA**. `index_add_` với duplicate index dùng atomic, có thể cho output khác nhau giữa run (race condition order).
2. **fp16/bf16 accumulation loss**. `index_add_` accumulate in-place ở dtype output (bf16). Sum nhỏ ở bf16 mất precision.
3. **Reshape+sum stable**. Sum qua dim cố định, ra fp32 intermediate, cast về output dtype cuối.

Trade-off: reshape+sum cần memory `(N, top_k, hidden)` temporary trước reduce. `index_add` dùng output buffer trực tiếp. Memory cost của reshape+sum cao hơn nhưng deterministic.

## Performance numbers (approximate)

GPU H100, batch_size=4, seq_len=2048, total N=8192, top_k=8, num_experts=64, hidden=4096, intermediate=14336.

| Backend | Latency / forward | Memory peak |
|---|---|---|
| Eager (for-loop) | 8.5 ms | OK |
| `batched_mm` | 4.2 ms | 1.6x of eager |
| `grouped_mm` (native) | 1.8 ms | 1.1x of eager |
| `grouped_mm` (fallback) | 5.1 ms | 1.1x of eager |

Numbers từ HF benchmark internal, approximate. Native `grouped_mm` thắng rõ ràng khi setup đầy đủ.

## Khi nào chọn backend nào

Decision tree:

```
PyTorch >= 2.9 và GPU SM80+ (Ampere)?
├── Yes -> grouped_mm
└── No
    ├── Need debug step-by-step? -> eager
    └── Production?
        ├── Batch lớn (>= 1024 token) -> batched_mm (vẫn nhanh hơn eager)
        └── Batch nhỏ -> eager (overhead grouped_mm không đáng)
```

Practical: hầu hết model HF default `grouped_mm` (set trong config khi `from_pretrained`). User chỉ override khi debug hoặc setup đặc biệt.

## Compile compatibility

`grouped_mm`:

- Native PyTorch 2.10+: compile OK (kernel registered as torch op).
- Fallback (custom op): compile OK (đã register fake + autograd).
- Sentinel handling: compile OK (`clamp_` và `masked_fill_` đều standard ops).

`batched_mm`:

- Compile OK với einsum (đã register).
- Sentinel handling: cũng OK.
- Memory overhead lớn hơn, có thể trigger recompile khi shape thay đổi.

Khuyến nghị: dùng `grouped_mm` + `torch.compile` cho production.

## Recompile risk

Với `grouped_mm`, `offsets` thay đổi mỗi batch (vì routing distribution khác). Compile mode có thể recompile mỗi forward nếu shape spec strict.

Mitigation:

- Set `dynamic=True` trong `torch.compile`.
- Hoặc pad offsets về fixed-size (nhưng memory waste).

Phần 4 chương 6 (training recipe) đi sâu hơn.

## Pitfall

**1. Quên check `_can_use_grouped_mm` trước khi set config**: model load OK, runtime crash vì kernel không có.

**2. Sentinel mask sai logic**: nếu mask before clamp, expert_ids out-of-range trigger error trước khi mask. Phải clamp trước, mask sau.

**3. Reshape+sum nhưng quên reshape**: `view(N, top_k, hidden).sum(dim=1)` cần `weighted.numel() == N * top_k * hidden`. Nếu N hoặc top_k sai, error khó debug.

**4. Sort permutation không inverse đúng**: nếu inv_perm không build đúng, output rows permute, model output sai.

**5. Hard-code `grouped_mm` ở config**: nếu deploy trên CPU hoặc old GPU, crash. Default nên auto-select.

Chương sau ta đi vào `load_balancing_loss_func`.
