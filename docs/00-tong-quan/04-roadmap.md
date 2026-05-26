---
title: Roadmap toàn series
---

# Roadmap toàn series

Chương này là bản đồ. Bạn không cần đọc tuần tự 41 chương theo thứ tự (mặc dù khuyến nghị). Mục tiêu là cung cấp đủ thông tin để bạn nhảy đến phần cần thiết.

## Tổng quan 5 phần

| Phần | Tên | Số chương | Mục tiêu |
|---|---|---|---|
| 0 | Tổng quan | 4 | Đặt nền: vì sao MoE, thuật ngữ, bản đồ |
| 1 | Foundations | 6 | Trực giác và math: router, routing, balancing, capacity, shared experts |
| 2 | HF MoE infrastructure | 5 | Walkthrough `integrations/moe.py`, `ExpertsInterface`, `grouped_mm` |
| 3 | Model walkthroughs | 12 | Đọc 10 model line-by-line + so sánh |
| 4 | Cross-cutting | 6 | EP, TP, quantization, serving, training |
| 5 | Comparison + decision guide | 5 | Bảng matrix, decision tree khi nào dùng gì |
| Resources | Glossary, cheatsheet, references | 3 | Tra cứu |

Tổng: 41 chương, ước lượng 45,000 từ.

## Đề xuất ba đường đọc

### Đường 1: Tuần tự (khuyến nghị cho người mới với MoE)

Phần 0 → Phần 1 → Phần 2 → Phần 3 (10 model theo thứ tự đã sắp) → Phần 4 → Phần 5. Đây là cách an toàn nhất, mọi khái niệm xuất hiện đúng thời điểm.

Thời gian ước lượng: ~15-20 giờ đọc nghiêm túc + ghi chú.

### Đường 2: Theo model cụ thể (cho người đã biết MoE cơ bản)

1. Đọc Phần 0 Chương 3 (thuật ngữ) nhanh.
2. Skip Phần 1 nếu đã nắm router/routing.
3. Đọc Phần 2 Chương 2-4 (HF infrastructure).
4. Nhảy đến Phần 3 Chương của model bạn quan tâm:
   - Cần baseline reference: Mixtral (Chương 2).
   - Cần state-of-the-art: DeepSeek-V3 (Chương 4).
   - Cần production quantization: GPT-OSS (Chương 6).
   - Cần hybrid kiến trúc: Jamba (Chương 9).
5. Bỏ qua Phần 4 nếu chỉ research; đọc Phần 4 nếu sắp deploy.

Thời gian ước lượng: ~5-8 giờ.

### Đường 3: Theo chủ đề cross-cutting

1. Phần 0 + Phần 1 Chương 4 (load balancing).
2. Phần 4 toàn bộ (EP, TP, quant, serving, training).
3. Phần 5 Chương 3 (load balancing comparison).
4. Quay lại Phần 3 chương model cụ thể nếu cần ví dụ.

Phù hợp cho: ML system engineer hoặc DevOps muốn serve MoE.

## Bản đồ chi tiết

### Phần 0: Tổng quan (4 chương)

- 01 Overview Phần 0 (chương đang đọc tiền nhiệm).
- 02 Vì sao Mixture of Experts.
- 03 Thuật ngữ cốt lõi.
- 04 Roadmap (chương này).

### Phần 1: Foundations (6 chương)

- 01 Overview Phần 1.
- 02 Router anatomy: gate logits, top-k, jitter, normalization.
- 03 Routing strategies: token-choice vs expert-choice, top-k variants, group routing.
- 04 Load balancing: aux loss, z-loss, aux-free bias adjustment.
- 05 Expert capacity và token dropping.
- 06 Shared experts và fine-grained partitioning.

### Phần 2: HF MoE infrastructure (5 chương)

- 01 Overview Phần 2.
- 02 `integrations/moe.py` anatomy (toàn bộ file 582 dòng).
- 03 `ExpertsInterface` và decorator `use_experts_implementation`.
- 04 `batched_mm` vs `grouped_mm`.
- 05 `load_balancing_loss_func` helper.

### Phần 3: Model walkthroughs (12 chương)

5 core:

- 02 Mixtral: baseline canonical, đọc kỹ làm reference cho mọi model sau.
- 03 Switch Transformers: encoder-decoder + top-1 + expert capacity.
- 04 DeepSeek-V3: aux-free + shared experts + group routing.
- 05 Qwen3-MoE: modern infra với `grouped_mm`.
- 06 GPT-OSS: production-grade với MXFP4 quantization.

5 supplementary:

- 07 OLMoE: fully-open recipe (Allen AI).
- 08 JetMoE: Mixture of Attention + Mixture of Experts cùng model.
- 09 Jamba: Mamba (SSM) + Transformer + MoE hybrid.
- 10 NLLB-MoE: translation, expert dropout, gating dropout.
- 11 PhiMoE: Phi-3.5-MoE, small-scale variant.

Wrap-up:

- 01 Overview Phần 3 (cách đọc).
- 12 Tổng kết so sánh 10 model.

### Phần 4: Cross-cutting (6 chương)

- 01 Overview Phần 4.
- 02 Expert parallelism (EP): `RouterParallel`, sentinels, all-to-all.
- 03 Tensor parallel với MoE: tương tác TP và EP.
- 04 Quantization MoE: MXFP4 (`mxfp4.py`), FP8 (`finegrained_fp8.py`).
- 05 Inference serving: continuous batching với MoE, vLLM hint.
- 06 Training recipe: megablocks, dropless, gradient through router.

### Phần 5: Comparison + decision guide (5 chương)

- 01 Overview Phần 5.
- 02 Routing comparison matrix.
- 03 Load balancing comparison.
- 04 Expert design comparison.
- 05 Khi nào dùng MoE vs dense (decision tree).

### Resources

- glossary.md (mở rộng từ Phần 0 Chương 3).
- cheatsheet.md (snippet copy-paste cho từng model).
- references.md (papers, code, blog).

## Phụ thuộc giữa các chương

Một số dependency cứng:

- Phần 1 Chương 2 (router) phải đọc trước Phần 1 Chương 3 (routing strategies).
- Phần 2 Chương 2-3 (infrastructure) nên đọc trước Phần 3 (model walkthroughs) nếu bạn muốn hiểu decorator `@use_experts_implementation`.
- Phần 3 Chương 2 (Mixtral) là baseline; các chương Phần 3 sau đó thường so sánh với Mixtral.
- Phần 4 Chương 2 (EP) cần Phần 3 Chương 2-5 (đã thấy các router design khác nhau).

Dependency mềm (có thể skip nếu vội):

- Phần 5 phụ thuộc Phần 3, nhưng chỉ chỗ cần tham chiếu cụ thể model.

## Cross-reference với repo trước

Một số khái niệm sẽ link đến chuỗi bài giảng `transformers-internals-foundation`:

- Attention backend (eager, SDPA, FlashAttention) khi đọc attention block của model MoE.
- KV cache khi nói về interaction giữa MoE và cache.
- `PreTrainedModel` conventions khi nói về `_tp_plan`, `from_pretrained`.

Khi gặp link, bạn có thể bỏ qua nếu đã quen, hoặc đọc nhanh chương được link để refresh.

## Lời khuyên thực hành

1. **Mở source code song song**. Mọi citation đều có path. Mở `transformers/src/transformers/...` cùng lúc với chương đang đọc.
2. **Ghi chú khi gặp pattern lặp lại**. Sau khi đọc 3 model, sẽ thấy pattern chung: `*Router` → `*Experts` → `*SparseMoeBlock`. Ghi xuống để nhận diện ở model thứ 4.
3. **Đừng cố nhớ mọi số config**. Bảng so sánh ở Phần 5 sẽ có. Tập trung vào "tại sao".
4. **Thử implement một mini-MoE**. Sau Phần 1, viết một SparseMoeBlock đơn giản từ đầu (10 dòng). Việc viết tay sẽ lộ ra chỗ chưa hiểu.

Phần 0 kết thúc. Sang Phần 1.
