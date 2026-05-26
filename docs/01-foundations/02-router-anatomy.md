---
title: Router anatomy
---

# Router anatomy

Router là module **nhỏ nhất nhưng quan trọng nhất** trong MoE. Một linear layer + softmax + topk, vài chục dòng code. Nhưng mọi quyết định thiết kế của MoE đều xoay quanh nó: phân kỳ load, độ ổn định, training stability, inference latency.

## Anatomy cơ bản: Mixtral

Đọc trực tiếp `MixtralTopKRouter`:

```python
class MixtralTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

(`src/transformers/models/mixtral/modeling_mixtral.py`, class `MixtralTopKRouter`.)

Forward có sáu bước. Đi từng bước:

**Bước 1: Flatten**

```python
hidden_states = hidden_states.reshape(-1, self.hidden_dim)
```

Input có shape `(batch, seq, hidden_dim)`. Router làm việc trên token level, không quan tâm batch/seq structure. Flatten thành `(batch*seq, hidden_dim)`. Gọi `B*S = N` là số token.

**Bước 2: Linear projection**

```python
router_logits = F.linear(hidden_states, self.weight)
```

`self.weight` shape `(num_experts, hidden_dim)`. Output shape `(N, num_experts)`. Đây là **gate logits**: mỗi token có một vector score, mỗi component là "affinity" với một expert.

Lưu ý: linear không có bias. Switch Transformer có bias. DeepSeek không có. Mỗi paper chọn riêng.

**Bước 3: Softmax normalize**

```python
router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
```

Softmax theo dim cuối (qua expert). Output là probability `(N, num_experts)`, sum = 1 mỗi token.

Lưu ý: `.float()` cast logits sang fp32 trước softmax. Đây là **selective precision**: dù model chạy bf16, router phải tính softmax ở fp32 để tránh underflow/overflow. Số bias ở bf16 chỉ có 8 bit mantissa, không đủ cho probability nhỏ. Switch Transformer paper section 2.2.4 nêu rõ điều này.

**Bước 4: Top-k selection**

```python
router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
```

Chọn `top_k` (= 2 cho Mixtral) expert có prob cao nhất. Output:

- `router_indices` shape `(N, top_k)`: ID của top-k expert.
- `router_top_value` shape `(N, top_k)`: prob của top-k expert.

**Bước 5: Renormalize**

```python
router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
```

Sau topk, `router_top_value` không còn sum = 1 (chỉ là top-k trong toàn bộ softmax distribution). Chia cho sum để renormalize. Output là probability trên top-k expert, sum = 1.

Lưu ý: bước này quan trọng vì khi combine output của expert, ta dùng weight này: `output = sum(weight_i * expert_i(x))`. Nếu weight không normalize, magnitude của output sẽ phụ thuộc routing entropy (high entropy: nhiều weight nhỏ, output nhỏ; low entropy: weight tập trung, output lớn). Training bị unstable.

**Bước 6: Return**

```python
return router_logits, router_scores, router_indices
```

Trả 3 thứ:

- `router_logits` (raw, chưa softmax): cần cho auxiliary loss (xem Chương 4).
- `router_scores` (= normalized top-k weights): dùng để combine expert output.
- `router_indices`: dùng để dispatch token đến expert.

## Jitter noise

Router có một trick training: nhân `hidden_states` với noise multiplicative trước khi forward.

```python
class MixtralSparseMoeBlock(nn.Module):
    def forward(self, hidden_states):
        ...
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        ...
```

`jitter_noise = 0.01` (default Mixtral 0.0). Mục đích:

1. **Exploration**. Encourage router thử expert khác nhau cho cùng một token, tránh stuck với một mapping cố định sớm trong training.
2. **Regularization**. Tương tự dropout, làm router không quá tự tin.

Chỉ active ở train (`if self.training`). Inference noise = 0.

Switch Transformer paper section 5.2 phân tích jitter, kết luận: noise nhỏ (0.01) cải thiện stability, noise lớn (0.1+) hại quality.

## Sigmoid thay softmax: DeepSeek-V3

DeepSeek-V3 dùng sigmoid thay softmax:

```python
class DeepseekV3TopkRouter(nn.Module):
    def forward(self, hidden_states):
        ...
        scores = F.linear(hidden_states, self.weight, None)
        scores = scores.sigmoid()  # <-- sigmoid, không softmax
        ...
```

(Trích đại ý từ `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`, class `DeepseekV3TopkRouter`.)

Khác biệt:

- **Softmax**: scores của các expert có constraint sum = 1. Cạnh tranh zero-sum.
- **Sigmoid**: mỗi expert có score độc lập trong `[0, 1]`. Không cạnh tranh.

Vì sao DeepSeek chọn sigmoid? Lý do trong paper (DeepSeekMoE):

1. **Top-k k lớn (k=8) với softmax** dẫn đến score nhỏ cho mỗi expert (sum=1 chia cho 256 expert), dễ underflow khi sort.
2. **Sigmoid cho phép multi-expert hot**: nhiều expert có thể có score cao đồng thời, phản ánh tốt hơn relevance độc lập.
3. **Bias adjustment dễ hơn**: cộng bias vào sigmoid không phá distribution như cộng vào softmax.

Hậu quả: sigmoid không cần renormalize sau topk (đã ở `[0,1]`), nhưng cần aux-free balancing khác (Chương 4).

## Group routing: DeepSeek-V3

DeepSeek-V3 thêm một lớp routing nữa: chia 256 expert thành `n_group = 8` nhóm (32 expert mỗi nhóm). Token chọn:

1. `topk_group = 4` nhóm có tổng score cao nhất.
2. Trong 4 nhóm đó, chọn `top_k = 8` expert có score cao nhất.

```python
# Pseudocode
scores = sigmoid(linear(hidden_states))  # (N, 256)
scores = scores.view(N, n_group, experts_per_group)  # (N, 8, 32)
group_scores = scores.topk(2, dim=-1).values.sum(dim=-1)  # (N, 8), tổng top-2 trong nhóm
selected_groups = topk(group_scores, topk_group, dim=-1).indices  # (N, 4)
# Mask out expert không thuộc selected_groups
masked_scores = scores * group_mask  # zero out các nhóm bị loại
top_k_indices = topk(masked_scores.view(N, 256), top_k, dim=-1).indices  # (N, 8)
```

Lý do hai-tầng routing:

1. **Communication locality**. Mỗi nhóm có thể được place trên một node (4 GPU). Token chỉ cần send đến `topk_group = 4` node thay vì tới mọi node. Giảm all-to-all bandwidth.
2. **Specialization granularity**. Nhóm có thể chuyên ngôn ngữ (group 1: tiếng Anh, group 2: code, group 3: math), expert trong nhóm chuyên hơn nữa.

Phần 1 Chương 3 sẽ vẽ rõ hơn flow này.

## Router dtype

Một số model cast router weight về fp32 explicitly để tránh sai số bf16:

```python
# Switch Transformer
self.dtype = getattr(torch, config.router_dtype)  # 'float32'
...
self.classifier = self.classifier.to(self.dtype)
router_logits = self.classifier(hidden_states)
router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(input_dtype)
```

(Trích từ `src/transformers/models/switch_transformers/modeling_switch_transformers.py`.)

Two layer dtype:

1. Linear projection ở fp32.
2. Softmax ở fp32.
3. Output cast về input dtype (bf16) trước khi return.

Mixtral không cast linear weight về fp32 (giữ bf16), nhưng cast softmax (`.float()`). Trade-off:

- Switch (full fp32): an toàn tuyệt đối, chậm hơn vài %.
- Mixtral (chỉ softmax fp32): nhanh hơn, đủ tốt cho top-2.

Cả hai approach đều OK trong practice.

## Số expert và config typical

| Model | num_experts | top_k | hidden_dim | router_params |
|---|---|---|---|---|
| Mixtral 8x7B | 8 | 2 | 4096 | 32,768 |
| Mixtral 8x22B | 8 | 2 | 6144 | 49,152 |
| Switch (large) | 128 | 1 | 1024 | 131,072 |
| DeepSeek-V3 | 256 | 8 | 7168 | 1,835,008 |
| Qwen3-30B-A3B | 128 | 8 | 2048 | 262,144 |
| GPT-OSS-20B | 32 | 4 | 2880 | 92,160 |
| GPT-OSS-120B | 128 | 4 | 2880 | 368,640 |
| OLMoE-1B-7B | 64 | 8 | 2048 | 131,072 |

Router params = `num_experts * hidden_dim`. So với một expert MLP (~`8 * hidden_dim^2` cho SwiGLU), router rất nhỏ (`<1%` của một expert). Cost của router là noise so với expert; không phải bottleneck.

## Pitfall

**1. Quên `.float()` cho softmax**: dùng bf16 cho softmax với 128+ expert gây loss precision, một số expert luôn được chọn vì noise floor.

**2. Quên renormalize top-k weights**: output magnitude phụ thuộc routing entropy, training unstable. Bug phổ biến khi tự implement.

**3. Jitter noise không tắt ở inference**: nondeterministic output, debug khó.

**4. Linear với bias cho router**: bias không cần thiết (linear tự học offset qua weight), thêm bias gây load imbalance khởi đầu.

**5. Init router weight với scale lớn**: gradient của router rất chậm so với expert (chỉ flow qua scalar weight nhân hidden_states). Init nhỏ (std=0.02) ngăn router thoái hoá ở step đầu.

Chương sau ta đi vào routing strategies (top-1, top-k, expert-choice, group).
