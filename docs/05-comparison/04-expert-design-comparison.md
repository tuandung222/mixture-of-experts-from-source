---
title: Expert design comparison
---

# Expert design comparison

Sau router và balancing, expert là quyết định thiết kế thứ ba. Chương này so sánh: coarse vs fine-grained, shared vs none, FFN ratio, activation choice.

## Coarse vs Fine-grained

| | Coarse-grained | Fine-grained |
|---|---|---|
| num_experts | ≤ 16 | ≥ 64 |
| expert FFN size | ≥ 8k | ≤ 4k |
| k/E ratio | high (12-25%) | low (3-12%) |
| Example | Mixtral, JetMoE | DeepSeek-V3, Qwen3, OLMoE |
| Specialization | Generalist | Specialist |
| Aux loss difficulty | Easy | Hard (more experts) |
| EP overhead | Low | High (all-to-all bigger) |

Xu hướng 2024+: fine-grained thắng.

**Reasoning**:

1. Specialization tốt hơn (256 expert × 1/32 size of dense ≠ 8 expert × 1/2 size).
2. Combinatorial routing rich hơn (C(256, 8) >> C(8, 2)).
3. Hardware (NVLink, grouped_mm) hỗ trợ fine-grained tốt.

## Shared expert

| Có | Không |
|---|---|
| DeepSeek-V2/V3, Granite-MoE-Shared | Mixtral, Switch, Qwen3, OLMoE, GPT-OSS, ... |

Shared expert luôn active. Học "general" knowledge. Routed expert "specialize".

**Pro**:

- Tách general vs specialized.
- Smooth output (shared luôn contribute baseline).
- Routed expert giảm capacity ngang (mỗi expert chỉ chuyên hẹp).

**Con**:

- Extra params luôn active → tăng active params.
- Replicate trên mọi EP rank (memory mỗi rank).
- Complexity code (riêng forward).

Verdict: shared expert tăng quality marginal (paper DeepSeek báo cáo +1-2%). Acceptable cho mọi scale ≥ 50B.

## FFN intermediate ratio

Dense Transformer (Llama, Mixtral): `intermediate / hidden ≈ 2.67-3.5x` (SwiGLU).

MoE:

| Model | hidden | intermediate | ratio |
|---|---|---|---|
| Mixtral | 4096 | 14336 | 3.5x |
| Switch base | 768 | 2048 | 2.67x |
| DeepSeek-V3 (routed) | 7168 | 2048 | 0.29x (small!) |
| DeepSeek-V3 (shared) | 7168 | 2048 | 0.29x |
| Qwen3-30B (moe) | 2048 | 768 | 0.38x |
| GPT-OSS (moe) | 2880 | 2880 | 1.0x |
| OLMoE | 2048 | 1024 | 0.5x |

Fine-grained model có ratio nhỏ (≤ 0.5x). Mỗi expert nhỏ hơn dense baseline.

**Tổng capacity** (qua mọi expert):

```
Mixtral: 8 expert × 3.5x = 28x dense FFN ratio.
DeepSeek-V3: 256 × 0.29x = 74x dense ratio (rất nhiều capacity).
Qwen3: 128 × 0.38x = 49x.
GPT-OSS: 128 × 1.0x = 128x.
```

Fine-grained có capacity total cao hơn coarse, dù mỗi expert nhỏ.

## Activation choice

| Function | Model |
|---|---|
| SwiGLU (silu × up) | Mixtral, DeepSeek, Qwen3, OLMoE, Jamba, PhiMoE |
| GELU approx (sigmoid GLU) | GPT-OSS |
| GELU | Switch (T5 base) |
| ReLU | Some old models |

SwiGLU dominant. GPT-OSS variant với clamp.

## Weight layout: 2D vs 3D

**2D layout (legacy)**:

```python
self.experts = nn.ModuleList([
    nn.Sequential(nn.Linear(d, d_ff), Act(), nn.Linear(d_ff, d))
    for _ in range(num_experts)
])
```

Switch, NLLB-MoE, Jamba dùng. Mỗi expert là một Module riêng. State_dict có `experts.0.weight`, `experts.1.weight`, ...

Pro: simple, debuggable per-expert.
Con: không tương thích `grouped_mm`. Cần loop Python.

**3D layout (modern)**:

```python
self.gate_up_proj = nn.Parameter(torch.empty(E, 2*d_ff, d))
self.down_proj = nn.Parameter(torch.empty(E, d, d_ff))
```

Mixtral, DeepSeek-V3, Qwen3, OLMoE, GPT-OSS, PhiMoE. Single 3D tensor.

Pro: contiguous memory, kernel-friendly, EP split.
Con: state_dict naming khác, debug khó hơn.

Verdict: 3D layout standard 2024+. Migration in progress cho legacy.

## Decorator pattern

3D layout + decorator:

```python
@use_experts_implementation
class MixtralExperts(nn.Module):
    ...
```

Decorator add flags (`has_gate`, `has_bias`, `is_transposed`, `is_concatenated`) và wrap forward.

Khi đọc model 2024+, check decorator để biết weight layout convention.

## Combine flag

GPT-OSS combine: `is_concatenated=False, is_transposed=True, has_bias=True`.

Pattern unique. Cần đọc kỹ chương 6 Phần 3.

Mọi model khác: default flags (`is_concatenated=True, is_transposed=False, has_bias=False, has_gate=True`).

## Decision tree expert design

```
Bao nhiêu total params?
├── < 10B
│   └── Coarse (8 expert, k=2), FFN ratio ~3.5x, no shared
│       Example: PhiMoE
├── 10-50B
│   └── Coarse-medium (8-32 expert, k=2-4), FFN ratio 2-3.5x, no shared
│       Example: Mixtral, GPT-OSS-20B
├── 50-200B
│   └── Fine (64-128 expert, k=4-8), FFN ratio 0.5-1.0x, optional shared
│       Example: Qwen3-MoE, OLMoE, GPT-OSS-120B
└── > 200B
    └── Ultra-fine (256+ expert, k=8), FFN ratio 0.3x, 1 shared
        Example: DeepSeek-V3
```

## Tổng kết

Modern MoE 2025 trends:

1. **Fine-grained > coarse**: hầu hết design mới đi theo.
2. **3D weight + decorator**: standard.
3. **Shared expert ổn**: optional improvement.
4. **SwiGLU dominant**: ngoại lệ GPT-OSS GELU.
5. **k/E ratio thấp (3-12%)**: sparsity cao.

Coarse-grained Mixtral pattern vẫn hợp lý cho:

- Small scale (≤ 50B).
- Simple ecosystem (fine-tune, LoRA, vLLM).
- Multi-vendor compat.

Fine-grained DeepSeek pattern tốt cho:

- Large scale (≥ 200B).
- Internal infra (custom kernel, EP cluster).
- Pushing quality boundary.

Chương cuối: decision tree MoE vs dense.
