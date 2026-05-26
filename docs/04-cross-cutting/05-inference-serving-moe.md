---
title: Inference serving MoE
---

# Inference serving MoE

Serve MoE production khác serve dense. Mỗi token đi qua subset expert khác nhau, batch handling phức tạp. Chương này phân tích continuous batching, vLLM/TGI, latency vs throughput trade-off.

## Single-stream inference

Single user, batch=1, decode mode:

- Mỗi step decode 1 token.
- Token đi qua `top_k` expert.
- Mỗi expert nhận 1 token (small workload).
- Compute mỗi expert: `(1, hidden) @ (hidden, intermediate)` matmul. GPU không full utilization.

Latency dominated by:

1. **Memory bandwidth** load weight expert active. Mỗi step load `top_k * expert_size` weight.
2. **Routing overhead**: argsort, dispatch, scatter. ~µs per layer.
3. **All-to-all** nếu EP. ~100µs per layer.

Throughput rất thấp. Single-stream MoE thường tệ hơn dense same active params.

## Continuous batching

Serve nhiều user đồng thời. Mỗi user ở phase khác (prefill, decode). Batch dynamic.

```
Time t:
  User A: prefilling 100 tokens
  User B: decoding token 50
  User C: decoding token 200
  User D: prefilling 30 tokens

Batched forward:
  Sequence A: tokens 0-99 (prefill)
  Sequence B: token 50 (decode, KV cache size 49)
  Sequence C: token 200 (decode, KV cache size 199)
  Sequence D: tokens 0-29 (prefill)

Total batch tokens: 100 + 1 + 1 + 30 = 132.
```

Each forward processes 132 tokens. Mixed prefill + decode.

MoE benefit:

- Mỗi expert nhận **nhiều token** (132 / 8 expert top-2 = ~33 token/expert).
- Matmul `(33, hidden) @ (hidden, intermediate)` decent GPU utilization.

Throughput tăng đáng kể so với batch=1.

## vLLM

vLLM (UC Berkeley) là serving engine open-source phổ biến. Hỗ trợ MoE:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,           # TP across 2 GPU
    enable_expert_parallel=True,       # EP (experimental)
    max_num_batched_tokens=8192,       # max batch size
    quantization="awq",                # optional quantize
)

prompts = ["Hello", "What is MoE?", ...]
params = SamplingParams(temperature=0.7, max_tokens=100)
outputs = llm.generate(prompts, params)
```

vLLM features:

1. **Paged KV cache**: KV cache management qua page, hỗ trợ continuous batching efficient.
2. **GPU memory utilization**: pack token vào batch tới target memory.
3. **Speculative decoding**: support draft + target model.
4. **MoE kernel optimize**: native CUDA kernel cho dispatch (faster than HF eager).

vLLM EP: gần stable. TP stable. Combined với continuous batching → state-of-the-art throughput cho MoE serving.

## TGI (Text Generation Inference)

HuggingFace's serving framework. Similar features:

```bash
docker run --gpus all -p 8080:80 ghcr.io/huggingface/text-generation-inference:latest \
  --model-id mistralai/Mixtral-8x7B-Instruct-v0.1 \
  --num-shard 4 \
  --quantize bitsandbytes-nf4
```

TGI:

- TP across `num-shard` GPU.
- Quantization options (bitsandbytes, GPTQ, AWQ, MXFP4).
- Continuous batching.
- HTTP API.

So với vLLM: TGI có UI/UX tốt hơn, vLLM throughput cao hơn (theo benchmark Mistral, vLLM 1.2-1.5x faster).

## Latency vs Throughput

Hai metric trade-off:

**Latency (TTFT, TBT)**:

- TTFT: Time to first token. Prefill cost.
- TBT: Time between tokens. Decode cost.
- Lower batch → lower latency per user.

**Throughput (TPS, RPS)**:

- TPS: Tokens per second across all users.
- RPS: Requests per second.
- Higher batch → higher throughput, but TBT increases.

```
Single user, batch=1:
  TPS ≈ 100 (rough)
  TBT ≈ 10ms

8 user, batch=8:
  TPS ≈ 600
  TBT ≈ 13ms per user (slight increase)

32 user, batch=32:
  TPS ≈ 1500
  TBT ≈ 21ms per user
```

MoE benefit từ batching nhiều hơn dense (expert dispatch amortize).

## KV cache MoE

KV cache không khác dense về thiết kế:

- Mỗi layer có KV cache.
- Size = `2 * num_kv_heads * head_dim * seq_len * batch * dtype_bytes`.

Khác biệt: với GQA + MLA, cache nhỏ hơn MHA.

**MLA (DeepSeek-V3)**: cache compressed latent, không K/V full. ~7x nhỏ hơn MHA standard.

```
Mixtral 8x7B với GQA 8 KV heads, head_dim 128:
  Cache per layer per token = 2 * 8 * 128 * 2 bytes = 4 KB.
  Cache 32 layer × 4096 token × batch 8 = 4 GB.

DeepSeek-V3 với MLA latent 512:
  Cache per layer per token = 512 * 2 bytes = 1 KB.
  Cache 58 MoE layer × 4096 × 8 = 1.9 GB. ~7x nhỏ.
```

MLA cho phép batch lớn hơn → throughput cao hơn.

## Speculative decoding với MoE

Speculative decoding: draft model (nhỏ, nhanh) gen `n` token, target model (lớn, slow) verify song song.

MoE cho target model: verification batch là `n` token, mỗi token chạy qua top-k expert. Routing overhead amortize.

Improvement: 2-4x throughput so với without speculative.

vLLM/TGI hỗ trợ. Draft model thường dense (faster), target MoE (slow but accurate).

## Cold start

MoE có cold start issue:

- Total params lớn (Mixtral 46GB, DeepSeek 1.3TB).
- Load model lần đầu chậm.
- Memory footprint cao dù chỉ subset expert active.

Mitigation:

- Quantization: load smaller weight.
- Lazy load expert: chỉ load expert được dùng (đột phá khi router uniform, không tốt khi skewed).
- Pre-warm: load model trước traffic.

## Production architecture

```
[Client] -> [Load Balancer]
              |
              v
       [Server Pool]
              |
   [Server 1: vLLM, 4 GPU]
   [Server 2: vLLM, 4 GPU]
   [Server 3: vLLM, 4 GPU]
              |
              v
    [Shared KV cache (Redis?)] (optional)
              |
              v
   [Model weight (S3/Disk)]
```

Mỗi server pool tự manage MoE inference. Load balancer route request.

## Cost analysis

GPT-OSS-120B production hosting (approximate, 2025):

- Hardware: 1× H100 80GB = $40k or $4/hr cloud.
- Throughput: 500 TPS (continuous batching).
- Cost per million tokens: ~$0.10 (compute only).

Dense 70B same quality (Llama-3-70B):

- Hardware: 1× H100 = same.
- Throughput: 200 TPS (smaller batch).
- Cost: ~$0.25 per million tokens.

MoE ~2.5x cheaper per token. Margin in production hosting business.

## Pitfall

**1. Batch quá nhỏ cho MoE**: batch=1 tệ. Cần batch ≥ 8 để MoE benefit kick in.

**2. KV cache fragmentation với continuous batching**: paged cache (vLLM) handle, naive cache không.

**3. Routing imbalance ở inference**: aux loss train không guarantee inference balance. Monitor expert utilization production.

**4. Speculative draft model**: phải share vocab với target. Mixtral draft cho Mixtral target OK; cross-architecture problematic.

**5. EP + continuous batching**: dispatch overhead per micro-step. Batch size phải đủ lớn để amortize.

**6. Long context (`>= 128k`) MoE**: KV cache đầy memory trước expert. Cần quantize KV (KIVI, HQQ).

Chương cuối Phần 4: training recipe.
