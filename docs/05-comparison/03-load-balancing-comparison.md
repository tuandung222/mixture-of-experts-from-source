---
title: Load balancing comparison
---

# Load balancing comparison

Cách balance expert đa dạng. Chương này đặt cạnh nhau ba paradigm: aux loss (Mixtral/Switch), z-loss (Switch/ST-MoE/OLMoE), bias adjustment (DeepSeek-V3). Cho mỗi paradigm: công thức, gradient flow, hyperparameter.

## 1. Auxiliary loss (Switch / Mixtral / hầu hết)

### Công thức

```
f_i = fraction of tokens routed to expert i
P_i = mean routing probability of expert i

L_aux = sum_i (f_i * P_i) * num_experts
```

Minimize khi cả f và P uniform: `f_i = P_i = 1/E` → loss = top_k (paper Switch eq 4).

### Gradient flow

`P_i` qua softmax → gradient flow router weight. `f_i` qua argmax (top-k), non-diff, no gradient.

Result: router weight được update từ P, encouraging uniform.

### Hyperparameter

```
router_aux_loss_coef:
  Mixtral, PhiMoE, Jamba, GPT-OSS, Qwen3-MoE: 0.001
  Switch: 0.01
  OLMoE: 0.01
  NLLB-MoE: 0.01
```

Coef nhỏ hơn (0.001) cho model coarse (k=2), lớn hơn (0.01) cho fine-grained hoặc capacity-based.

### Pros

- Đơn giản, một loss.
- Differentiable.
- Standard practice.

### Cons

- Conflict với task loss. Tune coef khó.
- Không guarantee perfect balance.
- Token-level statistics noisy.

## 2. Z-loss (Switch / ST-MoE / OLMoE)

### Công thức

```
log_z = log(sum_i exp(router_logits_i))
L_z = mean(log_z^2)
```

Phạt log-partition function large. Encourage router logits modest.

### Gradient flow

```
∂L_z / ∂logit_i = 2 * log_z * softmax(logit)_i
```

Reduces magnitude of large logits.

### Hyperparameter

```
router_z_loss_coef:
  Switch: 0.001
  OLMoE: 0.01
  ST-MoE: 0.001
```

Thường order of magnitude nhỏ hơn aux loss coef.

### Pros

- Stabilize numerical: prevent overflow, sharp distribution.
- Encourage exploration (entropy higher).

### Cons

- Side effect: smooth routing → less specialization initially.
- Tune extra coef.

### Khi cần

Z-loss đặc biệt hữu ích:

- Fine-grained (≥ 64 expert). Logit magnitude grow lớn khi expert nhiều.
- bf16 precision. Avoid overflow trong softmax.
- Long pretrain. Stability over time.

## 3. Bias adjustment aux-free (DeepSeek-V3)

### Công thức

```
score_i = sigmoid(router_logit_i)
score_for_choice_i = score_i + b_i   # b is the bias

# Choose top-k from score_for_choice
# Combine output using score_i (without b)
```

Bias `b_i` update qua callback:

```python
@torch.no_grad()
def update_bias_after_step(router, expert_load):
    expected_load = N * k / E
    for i in range(E):
        if expert_load[i] < expected_load:
            b_i += learning_rate_bias
        else:
            b_i -= learning_rate_bias
```

### No gradient

`b_i` không qua autograd. Update qua step rule (sign-based PID).

### Hyperparameter

```
learning_rate_bias: typically 1e-3 / num_experts
no aux coef (or very small for sequence-level only)
```

DeepSeek-V3 paper: `lr_bias = 0.001`, sequence-level aux coef `0.0001`.

### Pros

- Không conflict với task loss.
- Robust ở scale lớn (671B).
- Bias không méo output (chỉ dùng cho choice).

### Cons

- Cần implementation custom (callback ở optimizer step).
- Không native PyTorch flow.
- Convergence chậm hơn aux loss khi balance fail.

## So sánh ngang

| | Aux loss | Z-loss | Bias adjustment |
|---|---|---|---|
| Diff | Yes | Yes | No (rule-based) |
| Coef tune | Cần | Cần | Không |
| Stabilize numerical | No | Yes | N/A |
| Conflict task loss | Có | Một phần | Không |
| Implementation | Standard | Standard | Custom callback |
| Scale tested | All | Switch + OLMoE | DeepSeek 671B |

## Combine strategies

Real model thường combine:

**Mixtral / OLMoE / Qwen3 / GPT-OSS**:

```
total_loss = ce_loss + 0.001 * aux_loss (+ 0.01 * z_loss optional)
```

Mixtral chỉ aux. OLMoE + z-loss. GPT-OSS + jitter (deprecated 2024).

**DeepSeek-V3**:

```
total_loss = ce_loss + 0.0001 * sequence_aux_loss
+ bias_adjustment_callback_after_step()
```

Sequence aux loss phụ. Bias chính.

**Switch / NLLB-MoE**:

```
total_loss = ce_loss + 0.01 * aux_loss + 0.001 * z_loss
+ capacity_based_dropping_in_forward()
```

Capacity là hard constraint thay vì soft.

## Recommendation theo scale

### Small (≤ 50B total, ≤ 32 expert)

```
aux_coef = 0.001
z_loss_coef = 0 (không cần)
no bias adjustment
no capacity
```

Standard Mixtral recipe.

### Medium (50-200B, 64-128 expert)

```
aux_coef = 0.001
z_loss_coef = 0.001 (optional)
no bias adjustment
no capacity
```

Standard 2024 recipe. OLMoE, Qwen3, GPT-OSS pattern.

### Large (>200B, ≥ 256 expert)

```
aux_coef = 0.0001 (sequence-level only)
z_loss_coef = 0
bias_adjustment enabled
no capacity
```

DeepSeek pattern.

### Encoder-decoder translation

```
aux_coef = 0.01
z_loss_coef = 0.001
capacity_factor = 1.0-2.0
expert_dropout = 0.2
```

NLLB pattern.

## Pitfall

**1. Aux coef + bias adjustment cùng aggressive**: conflict, oscillate. Chỉ dùng một chính.

**2. Z-loss với sigmoid**: math nó cho softmax. Sigmoid không có log-partition meaningful. Skip.

**3. Bias update không sync giữa workers**: distributed train cần all-reduce expert_load count trước update bias. Else mỗi rank update khác → divergence.

**4. Aux loss noisy với batch nhỏ**: batch 1 → f_i là 0/1, P_i là 1 prob distribution. Loss noisy. Train với batch ≥ 32 (per-device).

**5. Capacity vs aux mâu thuẫn**: nếu force capacity, aux loss compute trên distribution post-drop. Stats không phản ánh router behavior thực. Practice: tính aux loss trước drop.

Chương sau ta đi expert design.
