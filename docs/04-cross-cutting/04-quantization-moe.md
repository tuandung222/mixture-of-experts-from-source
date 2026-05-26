---
title: Quantization MoE
---

# Quantization MoE

MoE total params lớn nhưng active params nhỏ. Quantize expert weight → memory bottleneck giảm rõ rệt. Chương này đọc MXFP4 (GPT-OSS native) và FP8 (DeepSeek-V3 train/serve).

## Vì sao quantize MoE

DeepSeek-V3 bf16: 1.34 TB. MXFP4: ~335 GB (4x giảm). Fit trên 4 H100 80GB.

Quantize **chỉ expert weight** (lớn) là đủ. Router, attention, embedding giữ precision cao (nhỏ, ảnh hưởng quality).

```
DeepSeek-V3 weight breakdown (approx bf16):
- Embed + lm_head: 1.8B params × 2 = 3.7 GB
- Attention (61 layers × MLA): 30B × 2 = 60 GB
- Router (61 × 256 × 7168): 1.1M × 2 = ~2 MB
- Shared expert (61 × 1 × 8M): ~500M × 2 = 1 GB
- Routed experts (61 × 256 × 17M): 264B × 2 = 528 GB
  -> chiếm ~95% memory

Nếu quantize routed expert MXFP4 (4-bit):
264B × 0.5 = 132 GB.
Total: ~200 GB. Fit 3 H100.

Nếu quantize FP8:
264B × 1 = 264 GB.
Total: ~330 GB. Fit 4 H100.
```

## MXFP4 (Microscaling FP4)

MXFP4 = Microscaling FP4. Open Compute Project standard (2023). Format:

- **Element**: 4-bit FP (E2M1: 1 sign, 2 exponent, 1 mantissa). Range 6 values: `[-6, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 4, 6]`.
- **Scale**: 8-bit FP (E8M0) shared cho block 32 element.
- **Effective bits**: `4 + 8/32 = 4.25 bits/element`.

Hardware support: NVIDIA Blackwell (B200) FP4 native tensor cores. Hopper (H100) emulation via FP8.

GPT-OSS native MXFP4. Hopper support qua kernel emulation.

## `integrations/mxfp4.py`

```python
# src/transformers/integrations/mxfp4.py
class Mxfp4Config(QuantizationConfigMixin):
    quant_method = QuantizationMethod.MXFP4
    block_size = 32
    modules_to_not_convert = None  # router, attention, lm_head, embed
    quant_type = "MXFP4"
    is_quantized = True
```

Config tương tự BitsAndBytesConfig, gptq, awq.

```python
class Mxfp4GptOssExperts(nn.Module):
    """Mxfp4 quantized version of GptOssExperts."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.intermediate_size = config.intermediate_size
        self.hidden_size = config.hidden_size

        # Weight stored as int8 (each int8 holds 2 FP4 values)
        # Total params / 2 for storage
        self.gate_up_proj_blocks = nn.Parameter(
            torch.empty((num_experts, hidden_size, intermediate_size, 16), dtype=torch.uint8),
            requires_grad=False,
        )
        # Scale stored separately as fp8
        self.gate_up_proj_scales = nn.Parameter(
            torch.empty((num_experts, hidden_size, intermediate_size // 32), dtype=torch.uint8),
            requires_grad=False,
        )
        ...
```

(Pseudocode based on `src/transformers/integrations/mxfp4.py`.)

Weight layout:

- `gate_up_proj_blocks`: int8 tensor, mỗi int8 chứa 2 giá trị FP4. Shape effectively `(E, in, out)` của FP4.
- `gate_up_proj_scales`: fp8 scale tensor, mỗi block 32 element có 1 scale.

Storage: 4-bit weight + (8/32) bit scale = 4.25 bits effective per param.

## MXFP4 forward

```python
def mxfp4_forward(hidden_states, gate_up_proj_blocks, gate_up_proj_scales):
    """Dequantize on-the-fly then matmul."""
    # Dequant weight to bf16
    weight_bf16 = dequant_mxfp4(gate_up_proj_blocks, gate_up_proj_scales)
    # Standard matmul
    return hidden_states @ weight_bf16
```

(Pseudocode.)

Pattern: weight stored low-bit, dequant on demand cho compute. Activation/intermediate giữ bf16.

Alternative trên Blackwell: matmul direct ở FP4 (FP4 tensor cores), no dequant overhead. Hopper (no FP4 cores): dequant + bf16 matmul.

## Quality cost MXFP4

Empirical (GPT-OSS paper):

- MXFP4 vs bf16: <0.5% quality drop trên benchmark.
- 4x memory saving.
- Latency similar (dequant overhead small).

Acceptable trade-off cho production.

## FP8 (E4M3, E5M2)

FP8 = 8-bit floating point. Hai variant chính:

- **E4M3**: 1 sign, 4 exponent, 3 mantissa. Range `[-448, 448]`. Better precision.
- **E5M2**: 1 sign, 5 exponent, 2 mantissa. Range `[-57344, 57344]`. Better range.

Use case typical:

- E4M3 cho weight và activation.
- E5M2 cho gradient (need range cho small grads).

Hardware: Hopper H100 native FP8 tensor cores. ~2x throughput so với bf16.

## `integrations/finegrained_fp8.py`

```python
# src/transformers/integrations/finegrained_fp8.py
class FineGrainedFP8Config(QuantizationConfigMixin):
    quant_method = QuantizationMethod.FINE_GRAINED_FP8
    activation_scheme = "dynamic"   # vs "static"
    weight_block_size = (128, 128)  # block size cho weight scale
    modules_to_not_convert = None
```

"Fine-grained": scale per-tile thay vì per-tensor. Tile size `(128, 128)`. Mỗi tile có 1 scale fp32.

Cost: scale tensor extra `(E, hidden/128, intermediate/128)`. Nhỏ so với weight.

Quality: fine-grained scale handle outlier tốt hơn per-tensor scale. Loss < per-tensor FP8.

## FP8 trong DeepSeek-V3

DeepSeek-V3 train với FP8 (E4M3). Paper:

- Weight: FP8 với fine-grained scale (block 128).
- Activation: FP8 dynamic scale (per-batch).
- Gradient: FP8 E5M2.
- Accumulator: BF16 hoặc FP32.

Stable training với 671B params. Memory cost 1/2 so với bf16. Throughput 1.5-2x.

## So sánh MXFP4 vs FP8

| | MXFP4 | FP8 |
|---|---|---|
| Bits | 4.25 | 8 |
| Memory saving | 4x bf16 | 2x bf16 |
| Hardware support | Blackwell native, Hopper emulate | Hopper native |
| Quality drop | ~0.5% | ~0.2% |
| Train support | Limited | Yes (DeepSeek) |
| Inference | Yes (GPT-OSS) | Yes |

MXFP4 aggressive hơn (more memory saving), quality drop slightly more.

## Quantize router?

Router weight nhỏ (`E × hidden`), không phải bottleneck memory. Quantize router:

1. Có thể bug: router quyết định route quan trọng, quantize gây route collapse.
2. Saving nhỏ.

Practice: **không quantize router**. `modules_to_not_convert = ["router"]`.

Tương tự embed_tokens và lm_head (`weight tied`, quan trọng).

## Quantize aware training (QAT)

GPT-OSS trained với MXFP4 simulation từ đầu (QAT). Pre-training với "fake quant" forward: dequant weight on-the-fly trong forward, gradient flow qua dequant function (straight-through estimator).

Cost: training slower (overhead simulate quant). Lợi: model học tolerate quant noise, deploy bf16 vs MXFP4 quality gần.

Mixtral, Llama quantize **post-training** (PTQ): pre-train bf16, quantize sau. Quality drop nhiều hơn.

## Memory and latency real numbers

GPT-OSS-120B với MXFP4:

- Weight: 60 GB (vs 234 GB bf16).
- KV cache 4096 context: 12 GB.
- Activation buffer: 8 GB.
- Total: ~80 GB. Fit 1 H100!

Latency (per token, single batch, H100):

- BF16: ~80 ms.
- MXFP4: ~85 ms (dequant overhead ~5%).

Memory bandwidth saving: 4x weight load → decode phase 3-4x faster với batch 1. Compute-bound prefill less impact.

DeepSeek-V3 với FP8 train:

- 4-node H100 cluster sufficient.
- BF16: 8 node minimum.

## Pitfall

**1. Quantize router**: route collapse. Always exclude.

**2. MXFP4 trên Hopper không có Blackwell**: emulate qua dequant + bf16 matmul. Slower than native. Future-proof when Blackwell mainstream.

**3. FP8 dynamic scale với batch nhỏ**: scale noisy, quality drop. Need batch ≥ 16 cho stable.

**4. Quantize trước tied weights**: lm_head tied với embed. Quantize lm_head → embed cũng quant. Phải exclude both.

**5. Mix precision EP**: nếu 1 rank quantize, 1 rank không, output không match. Phải uniform.

**6. Bias precision**: GPT-OSS có bias trong expert. Bias giữ bf16 dù weight MXFP4. Bias compensate quant error.

Chương sau ta đọc inference serving.
