---
title: Vì sao Mixture of Experts
---

# Vì sao Mixture of Experts

Trước khi đọc bất kỳ implementation nào, ta cần biết MoE giải bài toán gì. Bài toán đó không phải mới: scaling neural network bị giới hạn bởi FLOPs lúc inference. MoE là một câu trả lời cụ thể, không phải duy nhất, nhưng hiện đang chiếm ưu thế.

## Scaling laws: thước đo

Kaplan et al. 2020 và Hoffmann et al. 2022 (Chinchilla) công bố: loss của LLM giảm theo power law với compute, data, và parameter count. Đại khái:

```
loss ≈ A * compute^(-α) + B * data^(-β) + C * params^(-γ) + irreducible_loss
```

Để giảm loss, ta tăng compute và params đồng thời. Vấn đề: **inference cost tỉ lệ với active params, không phải total params**. Nếu mỗi forward pass dùng hết params, ta không có gì để giấu khi serve.

Dense Transformer dùng 100% params mỗi token. Phục vụ Llama-3-70B nghĩa là mọi token đều chạy qua 70B parameter. GPU memory bandwidth giới hạn token/giây khá nghiêm trọng (memory bound, không phải compute bound, ở batch nhỏ).

## Ý tưởng cốt lõi của MoE

Nếu **mỗi token chỉ thực sự cần một subset nhỏ của params**, ta có thể:

1. Tăng total params lên (capacity của model).
2. Active params mỗi token giữ nguyên (inference cost không đổi).

Đây là sparse activation. Một token "the" trong context tiếng Anh có thể không cần expert chuyên xử lý code Python; một token số trong tài liệu khoa học có thể không cần expert chuyên ngôn ngữ tự nhiên.

Cụ thể với Mixtral 8x7B:

- Total params: 46.7B (8 expert, mỗi expert ~5.6B, cộng với attention và embedding shared).
- Active params: 12.9B (top-2 trong 8 expert + attention + embedding).
- Inference cost: ~12.9B equivalent (chỉ 2/8 expert FFN được chạy).

Result: quality cỡ Llama-2-70B với inference cost cỡ Llama-2-13B (theo bench Mistral công bố). Đó là 5x improvement cho cùng inference budget.

## Vì sao bùng nổ 2023-2025

MoE không mới (Adaptive Mixtures of Local Experts, Jacobs et al. 1991). Vì sao đến gần đây mới bùng nổ?

**1. Hardware infrastructure đã sẵn**. EP (expert parallelism) cần all-to-all communication. NVLink, InfiniBand bandwidth đủ cao để chia expert giữa GPU mà không bottleneck. Mạng training cluster của 2023 khác xa 2017.

**2. Routing techniques chín muồi**. Switch Transformer (2021) và GShard (2020) giải bài toán load balance bằng auxiliary loss. ST-MoE (2022) thêm z-loss để stabilize. DeepSeek-V3 (2024) giải bằng bias adjustment không cần aux loss. Mỗi paper giải một góc.

**3. Software stack đã sẵn**. PyTorch `torch._grouped_mm` (2024) cho phép dispatch expert computation hiệu quả. HuggingFace `integrations/moe.py` chuẩn hoá pattern. vLLM, TensorRT-LLM support EP serving. Tooling 2024 không cản trở việc viết MoE.

**4. Có proof của Mistral**. Trước Mixtral 8x7B (December 2023), MoE chủ yếu là research curiosity. Mixtral release dưới Apache 2.0 với weight công khai. Cộng đồng download, fine-tune, deploy. Suddenly MoE không còn là academic, mà là production.

Sau Mixtral, DeepSeek, Qwen, OpenAI (rumored cho GPT-4 và GPT-5), Google (Gemini), Meta (rumored cho Llama-4), Anthropic (rumored cho Claude) đều theo hướng MoE.

## Khi nào MoE thắng dense

Không phải lúc nào MoE cũng tốt hơn. Một số điều kiện:

**MoE thắng khi**:

- **Total params lớn (`>= 30B`)**. Dưới scale này, overhead của routing và memory hệ thống ăn mất lợi ích.
- **Có infrastructure để serve EP** (multi-GPU, đủ bandwidth). Single-GPU MoE thường không có lợi.
- **Batch size lớn lúc inference**. Mỗi expert cần đủ token để fill compute. Batch nhỏ -> low utilization.
- **Workload đa dạng**. Nếu mọi query đều cùng kiểu, không có gì cho router phân biệt.

**Dense thắng khi**:

- **Model nhỏ (`<= 8B`)**. Active params đã đủ rẻ, không cần sparse.
- **Latency-critical với batch=1**. MoE routing có overhead, đặc biệt all-to-all communication.
- **Memory bound nghiêm trọng (edge device)**. MoE cần load toàn bộ total params lên RAM dù chỉ dùng subset; dense chỉ cần đúng những gì sẽ dùng.
- **Workload monoton**. Single-task fine-tune không tận dụng được phân kỳ expert.

Phần 5 Chương 5 sẽ có decision tree chi tiết. Bây giờ chỉ cần nhớ: **MoE không phải free lunch**. Trade-off cụ thể, và trade-off đó chỉ hợp lý ở một scale nhất định.

## Khái niệm "active params" và "total params"

Hai con số luôn đi cùng nhau khi nói về MoE:

- **Total params**: tổng số parameter trong model, bao gồm mọi expert dù được active hay không.
- **Active params**: số parameter thực sự được dùng cho **một token** mỗi forward pass.

Tỉ lệ `active / total` đo độ sparsity:

| Model | Active | Total | Tỉ lệ |
|---|---|---|---|
| Llama-3-70B (dense) | 70B | 70B | 100% |
| Mixtral 8x7B | 12.9B | 46.7B | 28% |
| Mixtral 8x22B | 39B | 141B | 28% |
| DeepSeek-V2 | 21B | 236B | 9% |
| DeepSeek-V3 | 37B | 671B | 5.5% |
| GPT-OSS-120B | 5.1B | 117B | 4.4% |

Xu hướng: tỉ lệ ngày càng thấp. DeepSeek-V3 và GPT-OSS đạt dưới 6%, nghĩa là 94% params không được chạm cho mỗi token. Đây là kết quả của thiết kế "fine-grained experts": nhiều expert nhỏ hơn là ít expert lớn. Phần 1 Chương 6 đi sâu.

## Một câu hỏi tự nhiên: vì sao không train dense model 671B?

Câu trả lời ngắn: **không phải vì không thể train, mà vì không thể serve**.

Dense 671B với bf16 chiếm 1.3 TB memory cho weight. Một H100 có 80 GB. Cần ít nhất 17 H100 để giữ weight, chưa kể activation, KV cache, batch. Latency mỗi token sẽ khủng khiếp vì mọi GPU phải sync mỗi layer.

DeepSeek-V3 với 671B total nhưng 37B active: với EP, mỗi GPU chỉ giữ subset expert. Một forward chỉ cần move data đến đúng GPU có expert được chọn. Total params lớn nhưng compute mỗi token tương đương 37B dense.

Nói cách khác: MoE biến **storage cost** (cần nhiều memory cho expert) thành **compute saving** (chỉ chạy subset). Trong cluster lớn, storage rẻ hơn nhiều so với compute. Đây là trade-off cho phép total params bùng nổ mà inference vẫn khả thi.

Chương sau ta liệt kê thuật ngữ.
