---
title: Khi nào dùng MoE vs Dense
---

# Khi nào dùng MoE vs Dense

Chương cuối, câu hỏi pragmatic: should you use MoE? Decision tree dựa trên scale, infrastructure, workload.

## TL;DR

**Dùng MoE khi**:

1. Total params target ≥ 30B.
2. Có infrastructure để serve (`>= 2` GPU, đủ memory).
3. Throughput ưu tiên hơn latency.
4. Batch size lúc serve ≥ 4.

**Dùng dense khi**:

1. Total params ≤ 10B.
2. Latency-critical batch=1.
3. Edge / single GPU memory bound.
4. Workload monoton (single domain fine-tune).

## Decision tree đầy đủ

```
Total params target?
├── ≤ 8B
│   └── Dense (Llama, Mistral, Phi-3). MoE overhead không đáng.
│
├── 8-30B
│   ├── Latency critical (batch=1, p50 < 50ms)?
│   │   └── Dense (Llama-2-13B, Mistral-7B).
│   └── Else (batch ≥ 4, throughput ưu tiên)?
│       └── MoE small (Mixtral 8x7B, PhiMoE, JetMoE).
│
├── 30-100B
│   ├── Single GPU 80GB?
│   │   └── Quantize MoE (Mixtral 4-bit) hoặc dense 70B 4-bit.
│   ├── 4-8 GPU?
│   │   └── MoE TP (Mixtral, Qwen3-30B).
│   └── Multi-node?
│       └── MoE + EP (DeepSeek-V2, Mixtral 8x22B).
│
└── ≥ 100B
    ├── Want best quality?
    │   └── MoE fine-grained (DeepSeek-V3, GPT-OSS-120B).
    └── Can't afford EP infra?
        └── Dense 70B với caveats hoặc quantize MoE 8-bit.
```

## Latency analysis

### Single batch, single GPU

```
Llama-2-70B (dense, 4-bit quantized):
  Memory: 40 GB.
  Decode latency: ~30 ms/token (H100).

Mixtral 8x7B (12.9B active):
  Memory: 26 GB bf16, 13 GB 4-bit.
  Decode latency: ~25 ms/token.
  (Active < Llama-70B, faster.)

DeepSeek-V3 (37B active):
  Memory: too big for single GPU bf16. Need MXFP4 + multi-GPU.
```

MoE thắng dense same active params về latency với batch nhỏ. Nhưng nếu compare same total params, dense ăn (mỗi token dùng hết).

### Batch 8, multi-GPU

```
4× H100 with TP=4:
  Llama-3-70B: throughput 1500 TPS.
  Mixtral 8x22B: throughput 2400 TPS.

  MoE thắng vì:
    - Active params nhỏ hơn (39B vs 70B).
    - Batch giúp expert utilization.
```

### Latency p99

MoE có higher latency variance:

- Routing distribution mỗi batch khác.
- Some expert hot (lag), some cold (idle).
- Tail latency phụ thuộc balance.

Dense: predictable, mọi token cost same.

Production trade-off: MoE ưu throughput, dense ưu p99.

## Cost analysis

Hosting cost (cloud, H100 $4/hr):

```
Llama-3-70B dense:
  Memory bf16 = 140 GB. Need 2 H100. Cost = $8/hr.
  Throughput ~300 TPS (batch 8).
  Cost per 1M token: $0.74.

Mixtral 8x22B MoE:
  Memory bf16 = 282 GB. Need 4 H100. Cost = $16/hr.
  Throughput ~1500 TPS.
  Cost per 1M token: $0.30.

DeepSeek-V3 (671B):
  Memory MXFP4 = 200 GB. Need 4 H100. Cost = $16/hr.
  Throughput ~800 TPS (large model, longer cache).
  Cost per 1M token: $0.55. Quality higher than Llama-3-70B.

GPT-OSS-120B:
  Memory MXFP4 = 60 GB. Need 1 H100. Cost = $4/hr.
  Throughput ~500 TPS.
  Cost per 1M token: $0.22.
```

GPT-OSS-120B economics tốt nhất. Memory low + competitive quality.

## Quality vs scale curve

Empirical (open benchmarks):

```
Score on MMLU (5-shot):
  Llama-3-8B:     65.0
  Mistral-7B:     63.0
  Mixtral 8x7B:   71.7   (Active 12.9B)
  Llama-3-70B:    80.5
  Mixtral 8x22B:  77.5   (Active 39B)
  DeepSeek-V3:    86.0   (Active 37B)
  GPT-OSS-20B:    65.0   (Active 3.6B)
  GPT-OSS-120B:   75.0   (Active 5.1B)
```

Quality cluster theo total params. MoE đạt quality dense same total với cost active thấp hơn.

Mixtral 8x7B (47B total) ≈ Llama-3-13B dense quality. Quality scaling theo total params chính (sparse activation efficiency ~70-80% of dense same total).

## Fine-tune trade-off

**LoRA Mixtral**:

- Memory: model 47B bf16 (94 GB) + LoRA adapter (~50 MB).
- Need 2 H100 80GB.
- Train time: ~10 hr cho 1B token.

**LoRA Llama-3-70B**:

- Memory: 140 GB + adapter.
- Need 2 H100.
- Train time: ~15 hr.

MoE fine-tune memory tương đương dense same total. Time MoE nhanh hơn (active params nhỏ).

**Full fine-tune MoE**:

- Memory: weight + gradient + Adam states. Adam 4x weight memory.
- 47B model × 16 bytes/param = 750 GB. Need cluster.
- Practical: chỉ LoRA cho MoE.

Dense 70B cũng full fine-tune khó. Practical chung: LoRA.

## Workload-specific

### Chat (general)

MoE thắng. Diverse query, expert specialization helps.

### Code generation

MoE thắng nếu có expert chuyên code. Mixtral, DeepSeek-Coder MoE benefit.

### Long context (≥ 128k)

Mamba-MoE hybrid (Jamba) compelling. Pure attention MoE: cache memory dominant, MoE hữu ích cho compute saving.

### Translation

MoE encoder-decoder (NLLB) đặc thù. Modern decoder-only MoE cũng OK cho translation (Qwen, Mistral).

### Edge inference

Dense small (≤ 7B). MoE total params bottleneck. Edge GPU ≤ 8GB không chạy MoE 30B+.

Exception: PhiMoE-mini (chưa ra) hoặc OLMoE-1B-7B với quant aggressive (4-bit) → 2GB. Edge khả thi.

### Multi-modal

MoE FFN trong vision-language hybrid model phổ biến (LLaVA-MoE, Janus). Same trade-off.

## Anti-pattern: MoE cho everything

Không phải lúc nào MoE cũng tốt:

**Anti-pattern 1**: Fine-tune dense 7B → migrate sang MoE 30B vì "MoE tốt hơn". Quality drop có thể vì:

- Pretrain data mismatch.
- MoE recipe khác.
- Aux loss tune.

Nên start với dense, migrate khi production cost justify.

**Anti-pattern 2**: Dùng MoE cho single-task fine-tune. Specialization phân kỳ giữa expert không hữu ích nếu task monoton. Dense + LoRA đủ.

**Anti-pattern 3**: Deploy MoE single GPU batch=1. MoE benefit từ batch + EP. Single GPU batch=1 chỉ làm overhead routing cost dominant.

## Future direction

Trends 2025-2026:

1. **Ultra-fine-grained**: ≥ 512 expert, k=8-16. Aggressive sparsity.
2. **Aux-free dominant**: bias adjustment trở thành standard.
3. **Per-expert quantization**: mỗi expert có scale riêng (already in fp8 fine-grained).
4. **MoE + Mamba/SSM hybrid**: Jamba pattern extend.
5. **MoE + reasoning**: chain-of-thought expert specialize.

Dense vẫn alive cho:

- Edge (Phi-3, Gemma, Llama small).
- Single-task production (specialize qua fine-tune).
- Research baselines.

## Decision summary

| Scenario | Choice | Why |
|---|---|---|
| Build chatbot 7B local | Dense (Llama-3-8B, Mistral) | MoE overhead không đáng |
| Build chatbot 30B host | MoE (Mixtral 8x7B) | Quality cao, cost giảm |
| Build coding agent serve 1000+ user | MoE (DeepSeek-V3, GPT-OSS-120B) | Throughput + quality |
| Fine-tune medical NER | Dense + LoRA (Llama, Mistral) | Single domain |
| Multi-task LLM platform | MoE (Mixtral, Qwen3) | Diverse query benefit |
| Edge mobile inference | Dense quantized (Phi-3-mini, Gemma) | Memory constraint |
| Research SOTA open | MoE (DeepSeek-V3) | Best quality |
| Translation 200 langs | MoE encoder-decoder (NLLB) | Specialize per language |

Phần 5 kết thúc. Resources (glossary, cheatsheet, references) còn lại.
