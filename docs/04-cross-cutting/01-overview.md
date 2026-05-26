---
title: Tổng quan Phần 4
---

# Phần 4: Cross-cutting concerns

Đến đây bạn đã đọc:

- Phần 0-1: Khái niệm và foundations.
- Phần 2: Infrastructure (`integrations/moe.py`).
- Phần 3: 10 model walkthrough.

Phần 4 mở rộng sang concern xuyên suốt model: distributed training/inference, quantization, serving, training recipe. Đây là phần "operational" nhất.

## Mục tiêu Phần 4

Sau Phần 4, bạn:

1. Hiểu expert parallelism (EP) hoạt động ra sao: dispatch, all-to-all, sentinel.
2. Biết tensor parallel (TP) và FSDP tương tác với EP.
3. Hiểu MXFP4 quantize MoE expert (GPT-OSS) và FP8 (DeepSeek-V3 fp8).
4. Hiểu continuous batching với MoE: vLLM, TGI.
5. Biết training recipe: aux loss tuning, capacity schedule, megablocks.

## Cấu trúc Phần 4

- Chương 2: **Expert parallelism (EP)**. RouterParallel, sentinels, all-to-all communication.
- Chương 3: **Tensor parallel với MoE**. TP plan declare, tương tác EP + TP.
- Chương 4: **Quantization MoE**. MXFP4 (`mxfp4.py`), FP8 (`finegrained_fp8.py`).
- Chương 5: **Inference serving**. Continuous batching, vLLM/TGI, latency vs throughput.
- Chương 6: **Training recipe**. Aux loss tune, megablocks kernel, gradient flow.

## Lý do tách Phần 4

Phần 3 đọc model một-một. Mỗi chương model có TP plan, quantization config, ... nhưng đề cập ngắn. Phần 4 là chỗ deep dive.

Khi deploy production:

1. **Single GPU**: Phần 1-3 đủ.
2. **Multi-GPU same node**: cần TP (Phần 4 Chương 3).
3. **Multi-GPU multi-node**: cần EP + TP (Phần 4 Chương 2-3).
4. **Latency critical**: cần quantization (Phần 4 Chương 4).
5. **Throughput critical**: cần continuous batching (Phần 4 Chương 5).
6. **Training**: cần aux schedule (Phần 4 Chương 6).

## Source code chính

```
src/transformers/integrations/
├── moe.py                              # Đã đọc Phần 2
├── mxfp4.py                            # MXFP4 quantization
├── finegrained_fp8.py                  # FP8 quantization
└── tensor_parallel.py                  # TP utilities

src/transformers/distributed/
├── tensor_parallel.py                  # TP plan apply
├── fsdp.py                             # FSDP wrap
└── utils.py
```

Walkthrough chi tiết các chương sau. Mỗi chương dẫn ra file cụ thể.

## Liên kết với chuỗi bài giảng nền

Nhiều khái niệm distributed (TP, FSDP, device_map) đã được cover ở chuỗi bài giảng `transformers-internals-foundation` Phần 5. Phần 4 này tập trung **specific cho MoE**.

Nếu chưa quen TP/FSDP cơ bản, đọc Phần 5 chuỗi đó trước, rồi quay lại.

Chương sau ta đi vào EP.
