---
title: Tổng quan Phần 3
---

# Phần 3: Model walkthroughs

Đến đây bạn đã có:

- Phần 0: Bản đồ tổng thể và thuật ngữ.
- Phần 1: Foundations - router, routing, balancing, capacity, shared/fine-grained.
- Phần 2: HF MoE infrastructure - `integrations/moe.py`, `ExpertsInterface`, `grouped_mm`.

Phần 3 là phần dài nhất: đọc 10 model line-by-line. Mỗi chương cố gắng follow cấu trúc giống nhau để bạn nhận diện pattern nhanh.

## Mục tiêu Phần 3

Sau Phần 3, bạn:

1. Đọc được bất kỳ `modeling_*moe*.py` mới mà HF release.
2. Phân biệt được design choice của mỗi model (top-k, num_experts, shared, aux loss, ...).
3. Biết model nào dùng infrastructure nào (eager loop legacy vs `@use_experts_implementation`).
4. Có thể fork một model và customize router cho task riêng.

## Cấu trúc mỗi chương

Mỗi chương model có sections:

1. **Context**: ai làm, năm nào, paper.
2. **Config key**: `num_experts`, `top_k`, `intermediate_size`, ... với giá trị actual.
3. **Router walkthrough**: class `*Router` hoặc `*Gate`, từng dòng.
4. **Experts walkthrough**: class `*Experts`, weight layout.
5. **SparseMoeBlock**: class wrap toàn module.
6. **Decoder layer**: cách MoE block integrate với attention.
7. **Đặc thù**: cái gì khác biệt so với model khác.
8. **Pitfall**: bug thường gặp.

## Thứ tự đề xuất

5 core đọc trước:

- **Chương 2**: Mixtral. Baseline. Đọc kỹ làm reference.
- **Chương 3**: Switch Transformers. Encoder-decoder + top-1 + capacity.
- **Chương 4**: DeepSeek-V3. State-of-the-art design.
- **Chương 5**: Qwen3-MoE. Modern infra.
- **Chương 6**: GPT-OSS. Production + MXFP4.

5 supplementary có thể đọc skim:

- **Chương 7**: OLMoE.
- **Chương 8**: JetMoE.
- **Chương 9**: Jamba.
- **Chương 10**: NLLB-MoE.
- **Chương 11**: PhiMoE.

Tổng kết:

- **Chương 12**: So sánh 10 model qua bảng.

## Pattern chung mà mọi model tuân theo

Sau khi đọc Mixtral, bạn sẽ thấy 9 model còn lại đều có cùng cấu trúc class:

```
modeling_X_moe.py
├── X_MoeMLP                  # MLP dùng cho shared expert hoặc dense layer
├── X_MoeExperts              # 3D weight tensor, @use_experts_implementation
├── X_MoeTopKRouter           # Linear + softmax/sigmoid + topk
├── X_MoeSparseMoeBlock       # Wrap router + experts
├── X_MoeAttention            # Standard MHA hoặc GQA
├── X_MoeDecoderLayer         # attn + (sparse_moe hoặc mlp)
├── X_MoePreTrainedModel      # Base, config_class, flags
├── X_MoeModel                # Body
├── X_MoeForCausalLM          # Task head
└── load_balancing_loss_func  # Helper
```

Khác biệt nằm ở chi tiết:

- Router: softmax vs sigmoid, có bias hay không, group routing hay không.
- Experts: concat vs not, transposed hay không, has_bias.
- SparseMoeBlock: có shared expert hay không, jitter noise hay không.
- DecoderLayer: tất cả layer là sparse hay alternate.

Bảng so sánh ngắn dẫn dắt:

| Model | num_experts | top_k | Router | Capacity | Shared | Special |
|---|---|---|---|---|---|---|
| Mixtral | 8 | 2 | softmax | no | no | Baseline |
| Switch | 32+ | 1 | softmax | yes (1.0-1.25) | no | T5 encoder-decoder |
| DeepSeek-V3 | 256 | 8 | sigmoid + group | no | 1 | aux-free bias |
| Qwen3-MoE | 128 | 8 | softmax | no | no | Modern infra |
| GPT-OSS | 32-128 | 4 | softmax | no | no | MXFP4 + clamp gate |
| OLMoE | 64 | 8 | softmax | no | no | Open recipe |
| JetMoE | 8 | 2 | softmax | no | no | MoA + MoE |
| Jamba | 16 | 2 | softmax | no | no | Mamba + MoE |
| NLLB-MoE | 128 | 2 | softmax | yes | no | Expert dropout |
| PhiMoE | 16 | 2 | softmax | no | no | Phi-3.5-MoE |

## Note cho người đọc skim

Nếu chỉ quan tâm 5 core, đọc chương 2-6 + 12. Skip 7-11.

Nếu chỉ quan tâm Mixtral (vì đó là model bạn đang fine-tune): đọc chương 2, rồi nhảy đến Phần 4 (cross-cutting).

Bạn cũng có thể đọc chương 12 đầu tiên để có bird-eye view, rồi quay lại đọc model cụ thể.

Chương sau bắt đầu với Mixtral.
