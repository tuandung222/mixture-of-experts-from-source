---
title: Tổng quan Phần 1
---

# Phần 1: Foundations

Phần 0 đã giới thiệu MoE ở cấp khái niệm. Phần 1 đi vào math và intuition của từng thành phần: router, routing strategy, load balancing, expert capacity, shared experts. Không có walkthrough code dài (đó là Phần 3). Mục đích Phần 1 là xây foundation đủ chắc để Phần 3 đọc nhanh.

## Mục tiêu Phần 1

Sau Phần 1, bạn:

1. Vẽ được forward pass của một SparseMoeBlock trên giấy.
2. Hiểu các lựa chọn thiết kế router: dimension, activation, normalization, jitter, group.
3. Phân biệt token-choice vs expert-choice routing, biết khi nào dùng cái nào.
4. Derive được auxiliary loss của Mixtral từ first principle.
5. Hiểu vì sao DeepSeek-V3 bỏ aux loss và dùng bias adjustment thay thế.
6. Biết khi nào dùng capacity factor + token dropping, khi nào dùng dropless.
7. Phân biệt shared expert (DeepSeek) và regular expert.
8. Biết "fine-grained" nghĩa là gì và vì sao xu hướng đi theo hướng đó.

## Cấu trúc Phần 1

Năm chương sau overview:

- Chương 2: **Router anatomy**. Gate logits, top-k selection, jitter noise, softmax vs sigmoid normalization, dtype tricks.
- Chương 3: **Routing strategies**. Token-choice (Mixtral, Switch) vs expert-choice (V-MoE), top-1 vs top-k, group routing (DeepSeek-V3).
- Chương 4: **Load balancing**. Auxiliary loss derivation, z-loss, bias adjustment aux-free.
- Chương 5: **Expert capacity và token dropping**. Switch paradigm, capacity factor, vs dropless variants 2024+.
- Chương 6: **Shared experts và fine-grained**. DeepSeek innovation, lý do fine-grained thắng coarse-grained.

## Pattern code tham chiếu

Để dẫn ý xuyên suốt Phần 1, ta dùng Mixtral làm baseline. Đây là `MixtralSparseMoeBlock` đầy đủ:

```python
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = MixtralTopKRouter(config)
        self.experts = MixtralExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
```

(Trích từ `src/transformers/models/mixtral/modeling_mixtral.py`, class `MixtralSparseMoeBlock`.)

Bốn thành phần ta sẽ giải thích trong Phần 1:

1. `self.gate` = router. Chương 2 đi sâu.
2. `jitter_noise` = exploration trick. Chương 2.
3. `top_k_weights, top_k_index` = output của router, đi đến dispatch. Chương 3.
4. `self.experts` = bộ expert. Forward của nó là chỗ load balancing matter (Chương 4) và capacity matter (Chương 5).

## Toàn cảnh forward pass MoE

```
Input: hidden_states (batch, seq, hidden_dim)
    |
    | [optional jitter noise nếu training]
    v
Reshape -> (batch*seq, hidden_dim)
    |
    v
Router (linear) -> router_logits (batch*seq, num_experts)
    |
    v
Softmax + Top-K -> top_k_weights, top_k_indices (batch*seq, k)
    |
    v
[Optional: normalize top_k_weights sum to 1]
    |
    v
For each expert in unique(top_k_indices):
    Gather tokens routed to this expert
    Run expert MLP
    Multiply output by routing weight
    Scatter back to output position
    |
    v
[Optional: add shared expert output]
    |
    v
Reshape -> (batch, seq, hidden_dim)
    |
    v
Output (+ optional router_logits for aux loss)
```

Mỗi bước có biến thể giữa các model. Bảng dưới đây là sneak peek:

| Bước | Mixtral | Switch | DeepSeek-V3 |
|---|---|---|---|
| Jitter noise | Có (train only) | Có | Không |
| Router weight init | Linear no bias | Linear with bias | Linear no bias |
| Normalization | Softmax | Softmax + selective precision | Sigmoid |
| Top-k | top-2 | top-1 | top-8 (trong nhóm) |
| Weight normalize | Sum to 1 | Không cần (k=1) | Sum to 1 |
| Aux loss | Có (1 loss) | Có (aux + z-loss) | Không (dùng bias) |
| Capacity | Không | Có (factor 1.0-1.25) | Không |
| Shared expert | Không | Không | Có (1-2 shared) |

Phần 1 sẽ giải thích từng cột. Phần 3 sẽ đọc code thực tế. Phần 5 sẽ so sánh ngang đầy đủ.

## Lý do tách Phần 1

Trước khi đọc 10 model, ta cần một mental model chung. Nếu nhảy thẳng vào model đầu tiên (Mixtral) mà chưa biết "auxiliary loss là gì", ta sẽ phải dừng đọc để search. Phần 1 cố gắng pre-load toàn bộ concept.

Một số chương Phần 1 sẽ overlap với Phần 3 và Phần 5. Đó là intentional. Phần 1 là **trực giác**, Phần 3 là **code**, Phần 5 là **so sánh**. Ba góc nhìn cho cùng một thứ. Sự lặp lại có chủ ý giúp ghi nhớ.

Chương sau ta vào router anatomy.
