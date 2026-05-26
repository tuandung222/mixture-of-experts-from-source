---
title: Routing strategies
---

# Routing strategies

Sau khi router xuất ra score, ta phải quyết định: token nào đi đến expert nào? Đây là **routing strategy**. Chương này so sánh các chiến lược: token-choice vs expert-choice, top-1 vs top-k, group routing.

## Token-choice vs Expert-choice

Hai paradigm cơ bản.

**Token-choice**: token chọn expert. Mỗi token có một router score vector, lấy top-k expert có score cao nhất. Đây là default cho hầu hết LLM MoE (Mixtral, Switch, DeepSeek, Qwen, ...).

**Expert-choice**: expert chọn token. Mỗi expert có một score vector qua tất cả token trong batch, lấy top-C token có score cao nhất. C là **capacity** per expert.

Cụ thể:

```python
# Token-choice (Mixtral)
router_scores = softmax(linear(hidden_states), dim=-1)  # (N, num_experts)
top_k_indices = topk(router_scores, k=2, dim=-1).indices  # (N, k) ID expert mỗi token
# Bao nhiêu token đến expert i? Đếm bằng cách count xuất hiện của i trong top_k_indices.
# Không xác định trước -> cần aux loss để balance.

# Expert-choice (V-MoE, EC-MoE)
router_scores = softmax(linear(hidden_states), dim=0)  # (N, num_experts), softmax over N
top_c_indices = topk(router_scores, C=N*k/num_experts, dim=0).indices  # (C, num_experts)
# Mỗi expert chọn đúng C token. Load balance tự động.
```

So sánh:

| Tiêu chí | Token-choice | Expert-choice |
|---|---|---|
| Load balance | Cần aux loss để encourage | Tự động (mỗi expert nhận C token) |
| Mỗi token được mấy expert | Cố định k | Biến động (0 to num_experts) |
| Inference với batch=1 | OK (token vẫn chọn được k) | Khó (single token không đủ để rank) |
| Token dropping | Có thể (vượt capacity) | Có thể (token không được expert nào chọn) |
| Causal masking decoder | OK | Khó (expert thấy "future" token để chọn) |
| Sử dụng phổ biến | Hầu hết LLM | V-MoE (vision), Expert-Choice paper |

**Vì sao LLM dùng token-choice?**

1. **Causal mask**. Expert-choice cần expert thấy toàn bộ batch để rank token. Decoder language model phải mask future. Không tương thích.
2. **Inference với batch nhỏ**. Khi serve LLM, batch có thể 1-8 token tại một thời điểm (mỗi user một câu hỏi). Expert-choice cần batch lớn để load balance ý nghĩa.
3. **Đơn giản**. Token-choice là natural extension của linear classifier.

**Vì sao vision dùng expert-choice?**

1. **Không causal**. ViT process tất cả patch song song, expert thấy toàn bộ image.
2. **Batch lớn ổn định**. Một image có 196-577 patch (ViT-Base), đủ để expert rank.
3. **Mỗi patch có importance khác nhau**. Expert chuyên patch quan trọng, ignore patch background.

Chuỗi này tập trung LLM. Từ chương này trở đi, mặc định token-choice. Phần 3 Chương 7 (NLLB-MoE) có nhắc expert-choice cho hoàn chỉnh.

## Top-1 vs Top-k

Trong token-choice, lựa chọn lớn nhất là **k**.

**Top-1**: mỗi token chỉ qua 1 expert. Switch Transformer là model lớn đầu tiên dùng top-1 với capacity factor. Lợi: rẻ inference (chỉ 1 expert / token), dễ phân tích. Hại: token bị tied vào một expert, không có "second opinion".

**Top-2**: mỗi token qua 2 expert, kết quả là weighted sum. Mixtral, JetMoE, NLLB-MoE, PhiMoE. Lợi: error correction (nếu expert 1 không chắc, expert 2 backup). Hại: cost 2x so với top-1.

**Top-k cao (k=4 đến 8)**: GPT-OSS (top-4), DeepSeek-V3 (top-8), OLMoE (top-8), Qwen3-MoE (top-8). Lợi: smooth output, mỗi token tích hợp kiến thức từ nhiều expert. Hại: cost k lần so với top-1.

**Quan trọng**: với fine-grained design (256 expert), top-k = 8 vẫn rẻ vì mỗi expert chỉ 1/32 size of dense MLP. So với top-1 của Switch (32 expert lớn), DeepSeek top-8 trong 256 expert có thể có **active params tương đương**.

Công thức ước lượng:

```
active_params_per_token = k * params_per_expert + shared_params
```

Với Mixtral 8x7B: k=2, params_per_expert ~ 5.6B, shared ~ 1.7B -> ~12.9B active.
Với DeepSeek-V3: k=8, params_per_expert ~ 4.1B, shared ~ 4.1B (2 shared expert) -> ~37B active.

Trade-off **k vs num_experts**:

| Design | k / num_experts | Tỉ lệ active/total |
|---|---|---|
| Coarse (Mixtral) | 2 / 8 = 25% | 28% |
| Medium (GPT-OSS-120B) | 4 / 128 = 3.1% | 4.4% |
| Fine (DeepSeek-V3) | 8 / 256 = 3.1% | 5.5% |
| Ultra-fine (OLMoE) | 8 / 64 = 12.5% | 16% |

Xu hướng 2024+: **k/num_experts tỉ lệ thấp** (3-5%) cho phép total params lớn mà active params giữ được nhỏ.

## Group routing

Khi num_experts lớn (`>=128`), top-k qua toàn bộ expert có hai vấn đề:

1. **Communication**. EP cần all-to-all giữa mọi expert location. 128 GPU -> 128² = 16384 cặp connection.
2. **Specialization**. Mỗi expert thấy ít token (1/num_experts) -> khó học pattern phức tạp.

DeepSeek-V3 giải bằng **group routing**: chia expert thành n_group nhóm, route hai tầng.

```python
class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok    # 8
        self.num_experts = config.n_routed_experts # 256
        self.n_group = config.n_group              # 8
        self.topk_group = config.topk_group        # 4
        ...

    def forward(self, hidden_states):
        # Step 1: Compute raw scores
        scores = F.linear(hidden_states, self.weight).sigmoid()  # (N, 256)

        # Step 2: Group-level scoring
        scores_grouped = scores.view(-1, self.n_group, self.num_experts // self.n_group)
        # (N, 8, 32)
        group_scores = scores_grouped.topk(2, dim=-1).values.sum(dim=-1)
        # (N, 8): tổng top-2 score trong mỗi group

        # Step 3: Pick top topk_group groups
        group_mask = ... # (N, 8) mask = 1 cho top-4 group, 0 còn lại

        # Step 4: Mask scores
        scores_for_choice = scores_grouped * group_mask.unsqueeze(-1)
        scores_for_choice = scores_for_choice.view(-1, self.num_experts)

        # Step 5: Top-k within allowed groups
        top_k_indices = scores_for_choice.topk(self.top_k, dim=-1).indices  # (N, 8)
        ...
```

(Pseudocode dựa trên `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`, class `DeepseekV3TopkRouter`. Code thực tế phức tạp hơn với bias adjustment, sẽ đi sâu Phần 3 Chương 4.)

Result: với 256 expert / 8 group / topk_group=4 / top_k=8, mỗi token chỉ "thấy" 128 expert (= 32 expert/group × 4 group selected), chọn 8.

Lợi:

1. **EP placement linh hoạt**. Mỗi group đặt trên một node. Token chỉ send đến 4 node (thay vì 8). Halving all-to-all bandwidth.
2. **Specialization theo group**. Có thể tune group size theo domain (1 group cho code, 1 cho math, ...).

Trade-off: complexity routing tăng. Code dài hơn, debug khó hơn.

## Routing với capacity (Switch paradigm)

Trước 2024, Switch Transformer và NLLB-MoE dùng **expert capacity**: mỗi expert được nhận **tối đa** `C = capacity_factor * (N * k) / num_experts` token. Token vượt capacity bị drop.

```python
class SwitchTransformersTop1Router(nn.Module):
    def forward(self, hidden_states):
        ...
        router_logits = self.classifier(hidden_states)
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
        router_logits, expert_index = torch.max(router_probs, dim=-1, keepdim=True)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Token priority: token đến trước được xử lý trước
        token_priority = torch.cumsum(expert_index, dim=-2)

        # Mask: token nào trong limit thì giữ, ngoài thì drop
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
        ...
```

(Trích từ `src/transformers/models/switch_transformers/modeling_switch_transformers.py`.)

Logic: nếu expert A đã nhận `C` token, token thứ `C+1` muốn đến A bị "drop" (chỉ qua residual, không qua expert). Tránh được all-to-all imbalance, vì payload mỗi expert cố định.

Lý do bỏ capacity (dropless) ở model 2024+:

1. **`grouped_mm` xử lý imbalance tốt**. Không cần force capacity, kernel tự handle variable-length.
2. **Token dropping hại quality**. Drop token đồng nghĩa expert bỏ thông tin.
3. **Aux loss đủ để balance**. Soft constraint thay hard capacity.

Phần 1 Chương 5 sẽ đi sâu trade-off.

## Routing in causal decoder

Một câu hỏi tự nhiên: với causal mask, expert thấy được token nào?

Trả lời: **router không có causal restriction**. Router là một linear layer, không có attention. Cho token `t`, router input là `hidden_states[t]`, output là score `(num_experts,)`. Score này không phụ thuộc token khác.

Vậy MoE decoder vẫn causal. Khác biệt:

- Attention: phải có causal mask, vì attention nhìn qua các token.
- Router: không cần (linear không cross-token).
- Expert FFN: pointwise, không cross-token.

Chỉ attention là chỗ duy nhất cần causal logic. MoE block hoàn toàn local theo token.

## Routing in encoder-decoder

Switch Transformer (T5-based) là encoder-decoder. Router có hai loại:

- **Encoder router**: trong encoder layer, route token theo content (toàn bộ sequence visible).
- **Decoder router**: trong decoder layer, route token theo content (causal, nhưng router vẫn local).

Cross-attention (decoder hỏi encoder K/V) không liên quan router. Router chỉ ở FFN block.

NLLB-MoE giống Switch: encoder-decoder, MoE block ở mọi layer hoặc subset.

## Pitfall

**1. Confuse routing strategy với load balancing**. Routing = quyết định dispatch. Load balancing = đảm bảo dispatch không lệch. Hai concern độc lập.

**2. Top-k với k=1 + capacity factor < 1.0**. Một số token chắc chắn bị drop. Quality issue. Switch dùng factor 1.0-1.25 để có margin.

**3. Group routing nhưng không có EP**. Lợi communication chỉ có nghĩa khi distributed. Single GPU thì group routing chỉ là overhead.

**4. Test routing với batch=1**. Lúc inference user-facing thường có batch=1. Aux loss training giả định batch lớn cho statistics. Inference có thể vào edge case (mọi token đến 1 expert).

**5. Quên train router cùng expert**. Router weight không được update sẽ stuck với init pattern. Cần ensure router weight có gradient flow (qua aux loss + expert output).

Chương sau ta đi vào load balancing, đặc biệt aux loss và bias adjustment.
