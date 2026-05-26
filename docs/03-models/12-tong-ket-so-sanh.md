---
title: Tổng kết so sánh 10 model
---

# Tổng kết so sánh 10 model

Sau khi đọc 10 model, chương này tổng hợp ngang qua các trục thiết kế chính. Là tài liệu tra cứu nhanh để bạn nhớ "Mixtral có shared expert không?" mà không phải mở chương riêng.

## Bảng tổng hợp config

| Model | Total | Active | Hidden | Layers | num_experts | top_k | Expert FFN | num_attn / num_kv |
|---|---|---|---|---|---|---|---|---|
| Mixtral 8x7B | 46.7B | 12.9B | 4096 | 32 | 8 | 2 | 14336 | 32 / 8 |
| Mixtral 8x22B | 141B | 39B | 6144 | 56 | 8 | 2 | 16384 | 48 / 8 |
| Switch-Base-128 | 7B | ~0.6B | 768 | 12+12 | 128 | 1 | 2048 | 12 / 12 |
| Switch-C-2048 | 1.6T | ~1B | 2080 | 15+15 | 2048 | 1 | 6144 | 32 / 32 |
| DeepSeek-V3 | 671B | 37B | 7168 | 61 | 256 routed + 1 shared | 8 | 2048 | MLA 128 |
| Qwen3-30B-A3B | 30B | 3B | 2048 | 48 | 128 | 8 | 768 | 32 / 4 |
| Qwen3-235B-A22B | 235B | 22B | 4096 | 94 | 128 | 8 | 1536 | 64 / 8 |
| GPT-OSS-20B | 21B | 3.6B | 2880 | 24 | 32 | 4 | 2880 | 64 / 8 |
| GPT-OSS-120B | 117B | 5.1B | 2880 | 36 | 128 | 4 | 2880 | 64 / 8 |
| OLMoE-1B-7B | 6.9B | 1.3B | 2048 | 16 | 64 | 8 | 1024 | 16 / 16 |
| JetMoE-8B | 8B | 2.2B | 2048 | 24 | 8 FFN + 8 attn | 2 + 2 | 5632 | 32 / 32 |
| Jamba-v0.1 | 52B | 12B | 4096 | 32 (mixed) | 16 | 2 | 14336 | 32 / 8 |
| NLLB-MoE | 54B | ~3B | 2048 | 24+24 | 128 | 2 | 8192 | 16 / 16 |
| PhiMoE | 42B | 6.6B | 4096 | 32 | 16 | 2 | 6400 | 32 / 8 |

## Routing & balancing

| Model | Router activation | Norm topk | Bias correction | Aux loss | Z-loss | Capacity | Jitter |
|---|---|---|---|---|---|---|---|
| Mixtral | softmax | yes | no | 0.001 | no | no | 0.0 (off) |
| Switch | softmax (fp32) | n/a (k=1) | no | 0.01 | 0.001 | yes (1.0-1.25) | 0.01 |
| DeepSeek-V3 | sigmoid | yes | yes (aux-free) | 0.0001 (seq-only) | no | no | no |
| Qwen3-MoE | softmax | configurable (default no) | no | 0.001 | no | no | no |
| GPT-OSS | softmax (local on top-k) | yes (implicit) | no | 0.001 | no | no | no |
| OLMoE | softmax | yes | no | 0.01 | 0.01 | no | no |
| JetMoE | softmax | yes | no | 0.01 (×2) | no | no | no |
| Jamba | softmax | yes | no | 0.001 | no | no | no |
| NLLB-MoE | softmax (fp32) | yes | no | 0.01 | 0.001 | yes (64) | 0.01 |
| PhiMoE | softmax | yes | no | 0.001 | no | no | 0.01 |

## Đặc thù kiến trúc

| Model | Shared expert | Fine-grained | Layer pattern | Attention type | Quant native |
|---|---|---|---|---|---|
| Mixtral | no | no | All MoE | GQA | no |
| Switch | no | no | sparse_step | T5 attention | no |
| DeepSeek-V3 | 1 | yes (256, fine) | 3 first dense, rest MoE | MLA | no (fp8 future) |
| Qwen3-MoE | no | yes (128) | All MoE (configurable) | GQA + QK norm | no |
| GPT-OSS | no | yes (32/128) | All MoE, alt sliding | GQA + sinks | MXFP4 |
| OLMoE | no | yes (64) | All MoE | MHA + QK norm | no |
| JetMoE | no | no | All MoE + MoA | MoA | no |
| Jamba | no | no | Mamba+Attn+MoE/MLP | Hybrid | no |
| NLLB-MoE | no | no | sparse_step in enc/dec | T5 attention | no |
| PhiMoE | no | no | All MoE | GQA + bias | no |

## Sparsity (active / total)

```
Tỉ lệ active params / total (smaller = more sparse):

GPT-OSS-120B:  4.4%   |█
DeepSeek-V3:   5.5%   |█
Qwen3-30B-A3B: 10%    |██
Qwen3-235B:    9.4%   |██
GPT-OSS-20B:   17%    |███
OLMoE:         18.8%  |████
Jamba:         23%    |█████
JetMoE:        27.5%  |██████
Mixtral 8x7B:  27.6%  |██████
Mixtral 8x22B: 27.7%  |██████
PhiMoE:        15.7%  |███
NLLB-MoE:      5.6%   |█
Switch:        ~9%    |██
```

Xu hướng 2023→2025: tỉ lệ ngày càng thấp. Fine-grained design dominate.

## Cấu trúc class chung

Nhìn lại pattern code:

```
Standard pattern (Mixtral, Qwen3, OLMoE, PhiMoE):
├── XExperts (@use_experts_implementation, 3D weight)
├── XTopKRouter (linear + softmax + topk)
├── XSparseMoeBlock (wrap router + experts)
└── XDecoderLayer (attn + sparse_moe_block)

DeepSeek-V3 variant:
├── DeepseekV3NaiveMoe (3D weight)
├── DeepseekV3TopkRouter (linear + sigmoid + bias buffer)
├── DeepseekV3MoE (router + naive_moe + shared_experts)
└── DeepseekV3DecoderLayer (first 3 dense, rest MoE)

Switch / NLLB-MoE variant (legacy):
├── XExperts (ModuleDict, one MLP per expert)
├── XTopKRouter (linear + softmax + topk + capacity)
├── XSparseMLP
├── XLayerFF (selectable sparse/dense)
└── XBlock (encoder hoặc decoder)

GPT-OSS variant:
├── GptOssExperts (@use_experts_implementation with flags, custom _apply_gate)
├── GptOssTopKRouter (linear + topk + local softmax)
├── GptOssMLP (wrap)
└── GptOssDecoderLayer (attn + mlp, alt sliding window)

JetMoE variant:
├── JetMoeMoA (attention experts)
├── JetMoeMoE (FFN experts)
├── JetMoeBlock (MoA attention + MoE FFN)

Jamba variant:
├── JambaSparseMoeBlock (ModuleList style)
├── JambaMambaDecoderLayer (Mamba + FFN)
├── JambaAttentionDecoderLayer (Attn + FFN)
└── (each FFN: MLP hoặc MoE theo period)
```

## Quyết định khi đọc / fork model mới

Khi gặp một model MoE mới:

1. **Identify pattern**: legacy (ModuleDict) hay modern (3D + decorator)?
2. **Read config**: num_experts, top_k, shared, capacity?
3. **Find router**: softmax hay sigmoid? Bias? Group routing?
4. **Find balancing**: aux loss coef? z-loss? bias adjustment?
5. **Find layer pattern**: all MoE hay alternate?
6. **Find attention**: MHA, GQA, MLA, MoA?

Mỗi quyết định map về một section trong Phần 1 hoặc Phần 5.

## Lời khuyên khi chọn model

**Nếu cần serve production**:

- Small/medium scale (`<= 30B`): Mixtral, Qwen3-30B, PhiMoE.
- Large scale với quant: GPT-OSS-120B (MXFP4 native).
- Best quality open: DeepSeek-V3 (cần infra lớn).

**Nếu cần fine-tune**:

- Mixtral: ecosystem mạnh nhất (tutorials, LoRA, vLLM support).
- Qwen3-MoE: modern infra, faster training.
- OLMoE: open recipe, có training code.

**Nếu cần research**:

- Switch: classic paradigm, paper foundational.
- DeepSeek-V3: state-of-the-art design.
- JetMoE: extension to attention.
- Jamba: hybrid architecture.

**Nếu cần edge**:

- PhiMoE: 6.6B active, smallest.
- OLMoE: 1.3B active, very small.

## Pattern nào sẽ thắng?

Không có câu trả lời tuyệt đối, nhưng xu hướng 2024-2025:

1. **Fine-grained over coarse** (256 expert hơn 8 expert).
2. **Top-8 over top-2** (smooth ouput).
3. **Sigmoid + bias adjust over softmax + aux** (DeepSeek influence).
4. **Shared expert phổ biến hơn** (general/specialized split).
5. **Dropless over capacity** (`grouped_mm` enable).
6. **`@use_experts_implementation` decorator** chuẩn hoá.

Mixtral pattern (2023) đang được thay bởi DeepSeek pattern (2024+). PhiMoE và Jamba còn theo Mixtral; Qwen3, GPT-OSS đã modernize.

Phần 3 kết thúc. Phần 4 sang cross-cutting concerns: EP, TP, quantization, serving, training.
