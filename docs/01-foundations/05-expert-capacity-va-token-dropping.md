---
title: Expert capacity và token dropping
---

# Expert capacity và token dropping

Switch Transformer (2021) đưa ra một solution gây tranh cãi cho load imbalance: **hard cap** số token mỗi expert nhận. Token vượt cap bị drop. Chương này phân tích vì sao technique này được dùng, vì sao bị thay thế bởi dropless ở model 2024+, và khi nào ta vẫn cần nó.

## Khái niệm capacity

Trong một batch có N token, top-k routing, E expert: kỳ vọng mỗi expert nhận `N * k / E` token. Nhưng router có thể lệch, một expert nhận nhiều, một expert nhận ít.

**Expert capacity**: số token tối đa một expert được phép nhận.

```
C = capacity_factor * (N * k / E)
```

- `capacity_factor = 1.0`: capacity = expected load. Nếu lệch, token vượt bị drop.
- `capacity_factor = 1.25`: capacity = 1.25x expected. Cho phép lệch ±25%, token vẫn được xử lý.
- `capacity_factor = 2.0`: capacity rộng, hầu hết token không bị drop, nhưng compute lãng phí.

Switch Transformer dùng 1.0-1.25. NLLB-MoE dùng 2.0 (conservative cho translation).

## Implementation: Switch Transformer

Đọc trực tiếp:

```python
class SwitchTransformersTop1Router(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.expert_capacity = config.expert_capacity
        ...

    def forward(self, hidden_states):
        ...
        router_logits = self.classifier(hidden_states)
        router_probs = nn.functional.softmax(router_logits, dim=-1, dtype=self.dtype).to(self.input_dtype)
        router_logits, expert_index = torch.max(router_probs, dim=-1, keepdim=True)
        expert_index = torch.nn.functional.one_hot(expert_index, num_classes=self.num_experts)

        # Mỗi token đi đến 1 expert (top-1)
        # Tính priority: token đến trước (theo thứ tự sequence) ưu tiên
        token_priority = torch.cumsum(expert_index, dim=-2)

        # Mask: token nào còn trong capacity thì giữ
        expert_capacity_mask = token_priority <= self.expert_capacity
        expert_index = expert_index * expert_capacity_mask
        router_probs = torch.max(router_probs, dim=-1).values.unsqueeze(-1)
        return router_probs, expert_index, router_logits
```

(`src/transformers/models/switch_transformers/modeling_switch_transformers.py`, class `SwitchTransformersTop1Router`.)

**Cumsum logic**:

```python
token_priority = torch.cumsum(expert_index, dim=-2)
```

Cumulative sum dọc theo dim sequence. Cho expert i, `token_priority[t][i] = 1 + 2 + ... + count(token đến i đến thời điểm t)`.

Ví dụ batch 1, seq 6, E=2:

```
expert_index:    [[1, 0], [1, 0], [0, 1], [1, 0], [1, 0], [0, 1]]   # top-1 expert mỗi token
token_priority:  [[1, 0], [2, 0], [2, 1], [3, 1], [4, 1], [4, 2]]   # cumsum
```

Nếu `expert_capacity = 3`:

```
mask = token_priority <= 3:  [[T,T], [T,T], [T,T], [T,T], [F,T], [F,T]]
```

Token 5 và 6 đến expert 0 bị drop (priority 4, 4 > 3). Token đến expert 1 vẫn OK (priority 2 < 3).

Token bị drop có `expert_index = 0` sau mask. Trong dispatch logic, expert_index = 0 nghĩa là không gửi đến expert nào (tương đương identity, chỉ qua residual).

## Vì sao có capacity?

Lý do thực dụng của Switch (2021):

1. **Static tensor shape**. JAX/XLA compile mong shape cố định. Mỗi expert nhận `C` token là static. Forward expert là `matmul (C, d) @ (d, d_ff)`, shape không đổi giữa batch. Compile nhanh.

2. **EP communication**. All-to-all dispatch yêu cầu mỗi expert nhận đúng `C` token để pad đồng nhất. Variable-length cần ragged tensor, lúc đó không có hardware support tốt.

3. **TPU friendly**. Switch train trên TPU. TPU yêu cầu shape static cho XLA compile.

Hạn chế:

1. **Drop token = drop information**. Nếu router lệch nặng (rich-get-richer), nhiều token bị drop. Quality drop.

2. **Capacity quá rộng = lãng phí**. Factor 2.0 nghĩa là expert được padded để giữ 2x expected. Một số slot empty, expert compute trên zero. Cost gấp đôi.

3. **Inference cost không deterministic**. Số token drop phụ thuộc routing distribution, hard to predict.

## Dropless: Mixtral và sau đó

Model 2024+ hầu hết bỏ capacity, dùng **dropless**: mọi token đều được xử lý.

```python
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""
    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states
```

(`src/transformers/models/mixtral/modeling_mixtral.py`, class `MixtralExperts`.)

Logic: loop qua các expert được hit, mỗi expert process tất cả token được gửi đến (không cap).

**Variable-length input** cho mỗi expert: expert A nhận 50 token, expert B nhận 30 token. Tensor shape không đồng nhất. PyTorch xử lý OK với for-loop. Compile-friendly khi dùng `grouped_mm`.

**Vì sao dropless khả thi 2024+?**

1. **`torch._grouped_mm` (PyTorch 2.9+)**: kernel matmul nhận `offs` chỉ ra boundary mỗi expert, xử lý variable-length hiệu quả. Phần 2 Chương 4 đi sâu.

2. **Megablocks (2023, Stanford/MosaicML)**: block-sparse matmul cho MoE. Mỗi expert có một block, kernel xử lý block-by-block.

3. **GPU compute đã đủ rẻ**. Cost của for-loop + index_add nhỏ so với matmul. Acceptable.

4. **Quality matter hơn**. Drop token là loss. Với scale tỉ đô, mỗi 0.1% quality đáng nghìn USD compute.

## Hybrid: NLLB-MoE

NLLB-MoE (translation) dùng capacity nhưng với factor cao + special expert dropout:

```python
# NLLB-MoE config
expert_capacity = config.expert_capacity  # ví dụ 2x expected
moe_token_dropout = 0.2                   # 20% expert bị drop ngẫu nhiên ở training
moe_eval_capacity_token_fraction = 1.0    # capacity factor at inference
```

Lý do:

1. Translation có sequence dài, batch nhỏ -> aux loss alone không đủ balance.
2. Expert dropout (drop entire expert) là regularization riêng (sẽ đi sâu Phần 3 Chương 10).
3. Eval với factor riêng để evaluate model robust.

Vẫn drop token, nhưng kết hợp với expert dropout làm router robust.

## Comparison: capacity vs dropless

| Tiêu chí | Capacity (Switch) | Dropless (Mixtral) |
|---|---|---|
| Token được xử lý | Some dropped | All |
| Tensor shape | Static (`C` per expert) | Variable |
| Compile-friendly | XLA tốt | Cần `grouped_mm` cho speed |
| Quality | Drop hurts | No drop |
| Inference latency | Predictable | Phụ thuộc routing |
| EP friendly | Yes (pad to C) | Cần all-to-all với offset |
| Phổ biến năm | 2021-2023 | 2024+ |

## "Capacity-aware" dropless: middle ground

Một số implementation hiện đại có capacity setting nhưng **không drop**, chỉ dùng để **pre-allocate buffer**:

```python
# Pseudocode
expert_capacity = ceil(N * k / E * 1.5)  # 1.5x cho margin
expert_buffer = torch.zeros(E, expert_capacity, hidden_dim)
# Dispatch token vào buffer, nếu vượt thì grow dynamic
# Không drop, chỉ realloc
```

PyTorch 2.x dynamic shape OK. Buffer pre-allocate giúp compile faster, nhưng nếu vượt thì grow thay vì drop.

Đây là pattern intermediate giữa Switch (drop) và pure dropless (no limit).

## Pitfall

**1. Capacity factor < 1.0**: chắc chắn drop token ngay cả khi router uniform. Tránh.

**2. Capacity factor quá lớn**: lãng phí compute. Mỗi expert làm matmul trên zeros.

**3. Confuse capacity với top-k**: capacity = số token mỗi expert nhận. Top-k = số expert mỗi token đi đến. Khác concept.

**4. Dropless không kiểm tra `expert_hit`**: expert không có token nào (expert_hit empty) phải skip, không run linear trên empty tensor.

**5. Static shape compile với dropless**: `torch.compile` recompile mỗi batch vì shape thay đổi. Cần dùng `grouped_mm` hoặc accept recompile.

**6. Drop token nhưng quên forward qua residual**: drop expert output không có nghĩa là zero out hidden states. Residual phải giữ. Switch code dùng identity (expert_index = 0 means no expert, residual flows).

Chương sau ta đi vào shared experts và fine-grained design.
