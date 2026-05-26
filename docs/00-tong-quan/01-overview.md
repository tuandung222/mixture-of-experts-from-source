---
title: Tổng quan Phần 0
---

# Phần 0: Tổng quan về Mixture of Experts

Phần 0 đặt nền cho toàn bộ chuỗi. Trước khi đọc bất kỳ `modeling_*moe*.py` nào, ta cần một cái khung khái niệm chung: MoE là gì, vì sao xuất hiện, từ vựng cốt lõi, và cách 10 model trong series này phân bố trong không gian thiết kế.

## Mục tiêu Phần 0

Sau Phần 0, bạn:

1. Phân biệt được dense Transformer và MoE Transformer ở mức kiến trúc.
2. Hiểu lý do MoE hấp dẫn dưới góc độ scaling laws và inference cost.
3. Đọc được các thuật ngữ: router, expert, top-k, gate, capacity, load balance, auxiliary loss, expert parallelism, shared expert, fine-grained.
4. Có bản đồ tổng thể 10 model sẽ đọc trong Phần 3, biết model nào đại diện cho paradigm nào.

## Cấu trúc Phần 0

Bốn chương:

- Chương 2: **Vì sao Mixture of Experts**. Scaling laws, sparse vs dense, lịch sử vắn tắt, lý do MoE bùng nổ 2023-2025.
- Chương 3: **Thuật ngữ cốt lõi**. Định nghĩa nhanh mọi từ ngữ sẽ gặp lặp lại.
- Chương 4: **Roadmap toàn series**. Cách đọc, thứ tự đề xuất, bản đồ 41 chương.

Phần 0 cố ý ngắn. Phần 1 sẽ deep dive vào router, routing, balancing. Phần 0 chỉ giúp bạn "biết tên" mọi thứ để Phần 1 trở đi không bị bỡ ngỡ.

## MoE Transformer ở góc nhìn 30 giây

Một decoder layer dense:

```
hidden_states = self_attention(hidden_states) + hidden_states
hidden_states = mlp(hidden_states) + hidden_states         # MLP = Feed Forward Network
```

Một decoder layer MoE thay `mlp` bằng `sparse_moe_block`:

```
hidden_states = self_attention(hidden_states) + hidden_states
hidden_states = sparse_moe_block(hidden_states) + hidden_states
```

Trong đó `sparse_moe_block` có dạng:

```python
def sparse_moe_block(hidden_states):
    router_logits = self.gate(hidden_states)        # (batch, seq, num_experts)
    selected, weights = topk(router_logits, k)      # chọn k expert cho mỗi token
    output = zeros_like(hidden_states)
    for expert_id in unique(selected):
        mask = (selected == expert_id)
        output[mask] += weights[mask] * self.experts[expert_id](hidden_states[mask])
    return output
```

Tất cả phần còn lại của Transformer (attention, layer norm, residual, embedding, lm_head) **không đổi**. Chỉ một module bị thay. Đây là điểm khiến MoE dễ tích hợp: lấy bất kỳ model dense nào, thay FFN bằng SparseMoeBlock là có một model MoE.

Câu hỏi thiết kế nằm ở chi tiết:

- Router lấy bao nhiêu expert mỗi token (top-1, top-2, top-8)?
- Bao nhiêu expert tổng (8, 64, 256)?
- Mỗi expert có MLP riêng hay shared parameters?
- Cân bằng load thế nào (auxiliary loss, bias adjustment, expert capacity)?
- Có expert luôn được dùng (shared expert) không?
- Khi inference distributed, expert nằm ở GPU nào (expert parallelism)?

Mỗi model trong Phần 3 chọn câu trả lời riêng. Phần 5 so sánh ngang.

## Một bảng nhanh 10 model

| Model | Routing | top-k | # experts | Shared | Aux loss | Đặc điểm |
|---|---|---|---|---|---|---|
| Mixtral | softmax + topk | 2 | 8 | không | có | Baseline canonical |
| Switch | softmax + top1 | 1 | 32-2048 | không | có + z-loss | Encoder-decoder + capacity |
| DeepSeek-V3 | sigmoid + topk | 8 | 256 | có | aux-free bias | SOTA design |
| Qwen3-MoE | softmax + topk | 8 | 128 | không | có | Modern infra |
| GPT-OSS | softmax + topk | 4 | 32-128 | không | có | MXFP4 quant |
| OLMoE | softmax + topk | 8 | 64 | không | có + z-loss | Open recipe |
| JetMoE | softmax + topk | 2 | 8 | không | có | MoA + MoE |
| Jamba | softmax + topk | 2 | 16 | không | có | Mamba + MoE |
| NLLB-MoE | softmax + topk | 2 | 128 | không | có + expert dropout | Translation |
| PhiMoE | sparsemax/softmax + topk | 2 | 16 | không | có | Small-scale |

Bảng này sẽ được giải thích kỹ ở Phần 3 và Phần 5. Bây giờ chỉ là teaser.

## Một số con số đáng nhớ

- **Mixtral 8x7B**: 46.7B total params, **12.9B active params** (top-2 trong 8 expert). Inference rẻ như model 13B nhưng quality gần model 70B.
- **DeepSeek-V3**: 671B total params, **37B active params**. Một trong những model open-weight mạnh nhất 2024.
- **GPT-OSS-120B**: 120B total params, **5B active params** (top-4 trong 128 expert). Tỉ lệ active/total dưới 5%.
- **Switch Transformer (Google, 2021)**: lên tới 1.6T total params, một trong những model lớn nhất từng được train ở thời điểm đó.

Tỉ lệ "active params / total params" thấp là dấu hiệu MoE đang tiến hoá theo hướng **fine-grained**: nhiều expert nhỏ hơn là ít expert lớn. Phần 1 và Phần 3 sẽ giải thích vì sao.

Chương sau ta giải thích lý do MoE bùng nổ.
