---
title: Routing comparison matrix
---

# Routing comparison matrix

Chương này tổng hợp design choice của router 10 model + paradigm tham khảo. Mục đích: bảng tra cứu khi designing MoE mới hoặc fork.

## Bảng đầy đủ

| Model | Norm | k | Bias | Group | Jitter | Capacity | Renorm topk | Note |
|---|---|---|---|---|---|---|---|---|
| Mixtral | softmax | 2 | no | no | optional (default 0) | no | yes | Baseline |
| Switch | softmax fp32 | 1 | no | no | 0.01 | yes (1.0-1.25) | n/a | T5 base |
| ST-MoE | softmax fp32 | 2 | no | no | 0.01 | yes (1.25) | yes | z-loss added |
| GShard | softmax | 2 | no | no | 0.01 | yes (2.0) | yes | Google T5 MoE |
| DeepSeek-V2 | softmax | 6 | yes (aux-free) | yes | no | no | yes | Pioneer aux-free |
| DeepSeek-V3 | sigmoid | 8 | yes (aux-free) | yes (8/4) | no | no | yes | SOTA design |
| Qwen2-MoE | softmax | 4 | no | no | no | no | yes | Older Qwen |
| Qwen3-MoE | softmax | 8 | no | no | no | no | configurable | Modern |
| GPT-OSS | softmax local | 4 | yes (linear) | no | no | no | yes (implicit) | Production |
| OLMoE | softmax | 8 | no | no | no | no | yes | Open recipe |
| JetMoE | softmax | 2 (×2) | no | no | no | no | yes | MoA + MoE |
| Jamba | softmax | 2 | no | no | no | no | yes | Hybrid Mamba |
| NLLB-MoE | softmax fp32 | 2 | no | no | 0.01 | yes (64) | yes | Translation |
| PhiMoE | softmax | 2 | no | no | 0.01 | no | yes | Small Phi-3.5 |
| V-MoE (vision) | expert-choice | n/a | no | no | no | yes | n/a | Image patches |
| Expert-Choice | reverse softmax | n/a | no | no | no | yes | n/a | Paper |

## Quyết định 1: softmax vs sigmoid

**Softmax (default)**:

- Mọi model trừ DeepSeek-V3.
- Constraint: sum = 1 across experts.
- Compete: chọn expert top → forced exclusivity.

**Sigmoid (DeepSeek-V2/V3)**:

- Mỗi expert độc lập `[0, 1]`.
- Multi-hot OK: nhiều expert có thể hot đồng thời.
- Cần `routed_scaling_factor` để magnitude match softmax.

**Khi nào sigmoid hợp lý**:

- Fine-grained (≥ 128 expert). Softmax với 256 expert → mỗi prob ~1/256, underflow rủi ro.
- Aux-free bias adjustment: dễ apply hơn (cộng bias không phá distribution sum=1).
- Top-k cao (≥ 8). Softmax topk weight cuối nhỏ; sigmoid mỗi weight đáng kể.

**Khi nào softmax đủ**:

- Coarse-grained (≤ 16 expert) + top-2.
- Pretrain truyền thống.

Xu hướng: model 2025+ chuyển sigmoid (DeepSeek influence).

## Quyết định 2: k = ?

| k | num_experts | Ví dụ | Trade-off |
|---|---|---|---|
| 1 | high (≥ 32) | Switch | Cheap inference, drop risk |
| 2 | low (8) | Mixtral, Jamba, PhiMoE | Classic, ổn định |
| 4 | medium (32-128) | GPT-OSS | Balance |
| 8 | high (≥ 64) | DeepSeek, Qwen3, OLMoE | Fine-grained smooth |

**Rule of thumb**: `k / num_experts ≈ 0.05 - 0.25`. Quá thấp expert starvation; quá cao approaching dense.

**Inference cost**: k × expert_size. Mixtral k=2 expert=14336 ≈ DeepSeek k=8 expert=2048 (same active FFN).

## Quyết định 3: group routing

Group routing chỉ useful khi:

- num_experts ≥ 64 (else 1 group enough).
- EP deployment (chia expert qua nhiều node).
- Want communication locality.

Implementation cost: code phức tạp hơn nhiều. DeepSeek-V3 worth it với 256 expert + multi-node.

Không cần group: Mixtral, OLMoE, Qwen3, GPT-OSS, ... Single-node hoặc TP-based.

## Quyết định 4: jitter noise

Switch + NLLB-MoE: 0.01.
Mixtral: default 0 (off).
Hầu hết model 2024+: không có.

Verdict: optional. Modern MoE với dropless + good aux loss không cần jitter.

## Quyết định 5: capacity vs dropless

Capacity (Switch, NLLB): drop token vượt cap.

- Pro: shape fix, EP all-to-all đồng nhất.
- Con: lose token info.

Dropless (Mixtral, DeepSeek, Qwen3, GPT-OSS): no drop.

- Pro: no info loss.
- Con: variable batch shape, cần `grouped_mm` kernel.

Verdict: dropless thắng cho LLM modern. Capacity còn trong NLLB cho translation specific.

## Quyết định 6: renorm topk weights

Mixtral, OLMoE, Qwen3 (configurable): renormalize.

Qwen3 default (no), GPT-OSS implicit (softmax over top-k only).

Rule: với softmax + topk, renorm thường an toàn (output magnitude stable). Sigmoid + bias không cần.

## Khi nào dùng group routing như DeepSeek-V3

Cần đồng thời:

1. ≥ 128 expert (group meaningful).
2. Multi-node deployment (locality matter).
3. Software stack support (HF, DeepSpeed).
4. Engineering effort acceptable (code phức tạp).

Else: skip group routing.

## Khi nào dùng sigmoid + aux-free

Cần đồng thời:

1. Fine-grained (≥ 128 expert).
2. Bias update callback implementation.
3. Sequence-level aux loss support.

Else: dùng softmax + standard aux loss (Mixtral style).

## Recipe đề xuất theo scale

**Tiny (1-10B total, < 16 expert)**:

- Router: softmax, no bias.
- k: 2.
- Aux loss: 0.001.
- No capacity.
- Example: PhiMoE.

**Small (10-50B total, 8-32 expert)**:

- Router: softmax, no bias.
- k: 2 hoặc 4.
- Aux loss: 0.001-0.01.
- No capacity.
- Example: Mixtral, GPT-OSS-20B.

**Medium (50-200B total, 32-128 expert)**:

- Router: softmax hoặc sigmoid.
- k: 4-8.
- Aux loss: 0.001 (+ optional z-loss).
- Renorm: yes.
- Example: Qwen3-MoE, OLMoE, GPT-OSS-120B.

**Large (> 200B total, ≥ 128 expert)**:

- Router: sigmoid với bias adjustment.
- k: 8.
- Group routing nếu multi-node.
- Aux loss: sequence-level + bias.
- Shared expert: 1-2.
- Example: DeepSeek-V3.

## Pitfall

**1. Top-1 với ít expert**: cheap nhưng quality drop drastic. Nếu k=1 thì num_experts phải ≥ 32 (Switch).

**2. Sigmoid không scale**: output magnitude nhỏ. Cần `routed_scaling_factor` (DeepSeek 2.5).

**3. Group routing với batch=1 (decode)**: chỉ 1 token, group selection noisy. Inference cần warm-up.

**4. Aux loss + bias adjustment**: dùng cả hai có thể conflict. DeepSeek-V3 dùng bias chính + sequence aux phụ.

**5. Renorm topk với sigmoid**: không sum=1 anyway, renorm meaningless. Bỏ.

Chương sau ta đi sâu load balancing comparison.
