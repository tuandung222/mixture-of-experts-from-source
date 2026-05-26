---
title: Load balancing
---

# Load balancing

Router học bằng gradient descent. Nếu ta không can thiệp, router sẽ thoái hoá: chọn 1-2 expert phổ biến cho mọi token, các expert khác bị bỏ. Đây là **expert collapse**. Chương này đi vào ba kỹ thuật chống collapse: auxiliary loss (Mixtral, Switch), z-loss (Switch), bias adjustment aux-free (DeepSeek-V3).

## Tại sao router thoái hoá

Gradient flow qua router:

```
Loss = L(model_output)
∂Loss/∂router_weight = ∂Loss/∂output · ∂output/∂routing_weight · ∂routing_weight/∂router_weight
```

Mỗi token đẩy router theo hướng "expert đang chọn cho tôi đúng". Nếu expert A đúng tốt cho token type X, gradient đẩy router cho mọi token gần X về phía A. Self-reinforcing: A xử lý nhiều token X -> A học X tốt hơn -> router chọn A cho X nhiều hơn -> ...

Sau vài bước, một số expert dominate. Phần còn lại nhận ít token, học chậm, ít được chọn. Equilibrium: 1-2 expert được chọn 90%+ thời gian. Còn lại là dead weight.

Đây là vấn đề **rich-get-richer**. Cần soft constraint hoặc hard constraint để ngăn.

## Auxiliary loss: Mixtral & Switch

Ý tưởng: thêm một loss term phụ phạt phân bố không đều.

**Notation**:

- N: số token trong batch
- E: num_experts
- k: top-k
- `s_i^t`: routing score (softmax output) của token t đến expert i
- `1[i ∈ topk(t)]`: indicator, =1 nếu expert i nằm trong top-k của token t

**Định nghĩa**:

- `f_i = (1/N) * sum_t 1[i ∈ topk(t)]`: fraction of tokens routed to expert i.
- `P_i = (1/N) * sum_t s_i^t`: average routing probability for expert i.

**Aux loss** (Switch Transformer paper, eq 4):

```
L_aux = E * sum_i (f_i * P_i)
```

Nhân với E để loss = 1 khi phân bố uniform (mỗi expert nhận 1/E token, có avg prob 1/E, total = E * (1/E) * (1/E) * E = 1).

**Vì sao loss này phạt imbalance?**

Cauchy-Schwarz: với sum constraint, `sum(f_i * P_i)` minimize khi `f_i` và `P_i` đồng đều. Imbalance cao -> term cao -> loss cao -> gradient đẩy router về uniform.

**Vì sao dùng cả f và P?**

Chỉ dùng f (tỉ lệ token): gradient không flow back đến router vì f là argmax (top-k), không differentiable. Cần P (prob, differentiable).

Chỉ dùng P: P là continuous, có gradient, nhưng không phản ánh exact distribution token. Có thể router output entropy uniform (P uniform) nhưng topk vẫn lệch.

Cả hai: `f_i * P_i` differentiable qua P, đo lường thực tế qua f. Best of both.

## Implementation thực tế

```python
def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    if isinstance(gate_logits, tuple):
        compute_device = gate_logits[0].device
        concatenated_gate_logits = torch.cat([layer_gate.to(compute_device) for layer_gate in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
    router_prob_per_expert = torch.mean(routing_weights, dim=0)

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
```

(Lược trích từ `src/transformers/models/mixtral/modeling_mixtral.py`, function `load_balancing_loss_func`.)

Bước theo notation:

1. `concatenated_gate_logits`: gộp gate logits qua mọi layer. Shape `(L*N, E)`.
2. `routing_weights = softmax(...)`: shape `(L*N, E)`, mỗi token một prob distribution.
3. `selected_experts = topk(...)`: shape `(L*N, k)`, expert ID.
4. `expert_mask = one_hot(...)`: shape `(L*N, k, E)`, 1 ở vị trí selected.
5. `tokens_per_expert = mean(expert_mask)`: shape `(k, E)`, hoặc `(E,)` sau mean qua k. Tương đương `f_i`.
6. `router_prob_per_expert = mean(routing_weights)`: shape `(E,)`. Tương đương `P_i`.
7. Loss = sum(f * P) * E.

Trong code Mixtral, có aux loss applied khi `output_router_logits=True` (training mode):

```python
class MixtralForCausalLM(...):
    def forward(self, ..., output_router_logits=None, ...):
        ...
        if labels is not None:
            loss = self.loss_function(logits, labels, ...)
            if output_router_logits:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits, self.config.num_local_experts, self.config.num_experts_per_tok, attention_mask
                )
                loss += self.config.router_aux_loss_coef * aux_loss
```

Coef typical 0.001-0.01. Quá nhỏ -> imbalance vẫn xảy ra. Quá lớn -> router học balance over task accuracy.

## Z-loss (Switch Transformer)

Thêm một loss thứ hai phạt **large logit magnitude**:

```
L_z = (1/N) * sum_t (log sum_i exp(x_i^t))^2
```

trong đó `x_i^t` là raw router logit. `log sum_i exp(x_i^t)` là log-partition function của softmax. Nếu logit lớn, term lớn, loss lớn.

**Vì sao cần?**

Aux loss + softmax có thể dẫn đến router_logit lớn (để boost probability mass của expert chọn). Logit lớn gây:

1. Numerical issue ở bf16.
2. Sharp probability, nhỏ entropy, less exploration.

Z-loss giữ logit modest. Coef nhỏ (Switch dùng 0.001).

```python
# Switch z-loss
log_z = torch.logsumexp(router_logits, dim=-1)
z_loss = (log_z ** 2).mean()
```

Mixtral không có z-loss. ST-MoE, Switch, OLMoE có.

## Aux-free balancing: DeepSeek-V3

Aux loss có vấn đề: gradient của nó conflict với task loss. Router phải tradeoff balance vs task accuracy. Coef phải tune.

DeepSeek-V3 đề xuất **aux-free**: dùng bias adjustment dynamic.

**Ý tưởng**: thay vì train router weight để balance, ta thêm **bias** vào router logit. Bias không qua gradient, mà update theo simple rule:

```
bias_i ← bias_i + lr * sign(expected_load - actual_load_i)
```

- `expected_load = N * k / E`: token mỗi expert kỳ vọng.
- `actual_load_i`: token expert i thực sự nhận.

Nếu expert i underutilized (actual < expected), `bias_i` tăng -> logit i tăng -> chọn nhiều hơn ở step sau.

Code DeepSeek-V3:

```python
class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        ...
        self.e_score_correction_bias = nn.Parameter(torch.empty((self.num_experts,)))
        # Bias này không train qua gradient task loss
        # Update qua callback ở training step

    def forward(self, hidden_states):
        scores = F.linear(hidden_states, self.weight, None).sigmoid()  # raw scores
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)  # add bias
        # ... topk, gating ...
```

(Trích đại ý từ `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`.)

**Khác biệt then chốt**:

- Bias là **detached parameter** ngoài graph gradient. Không có `loss.backward()` flow qua bias.
- Bias update qua **callback ở optimizer step**, không qua autograd.
- Khi combine score để weight expert output, dùng **score gốc** (không cộng bias), tránh bias làm méo output.

Pseudocode update bias:

```python
@torch.no_grad()
def update_bias_after_step(router, expert_load):
    expected = N * k / E
    for i in range(E):
        if expert_load[i] < expected:
            router.bias[i] += lr_bias
        elif expert_load[i] > expected:
            router.bias[i] -= lr_bias
```

**Lợi**:

1. Không conflict với task loss. Bias xử lý balance, weight xử lý task.
2. Không cần tune aux coef. Bias update rule đơn giản.
3. Stable hơn ở train scale lớn (DeepSeek-V3 dùng cho 671B model).

**Hại**:

1. Cần implement bias update callback, không native PyTorch.
2. Bias init quan trọng (start = 0 OK, nhưng nếu init lớn, lệch lệnh khó hồi).

DeepSeek paper "Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts" (2024) đề xuất technique này, thấy improvement nhỏ nhưng consistent.

## Sequence-level vs Batch-level

Một chi tiết: load balance tính trên scope nào?

**Batch-level**: tính f, P trên toàn batch (N = batch_size * seq_len). Mixtral, Switch default.

**Sequence-level**: tính f, P trên mỗi sequence riêng (N = seq_len). DeepSeek-V3 có một loss bổ sung gọi là "sequence-level balance loss".

Vì sao? Imagine một batch có 2 sequence rất khác nhau (1 code, 1 text). Batch-level balance OK (mỗi expert nhận đều khi cộng cả 2 sequence). Nhưng mỗi sequence có thể lệch (sequence code đi expert A, sequence text đi expert B).

Sequence-level loss phạt mỗi sequence lệch. Hậu quả: router học phân bố đều cả khi sequence đồng nhất content.

DeepSeek-V3 dùng **cả** batch-level (aux-free via bias) + sequence-level (regular aux loss với coef nhỏ).

## So sánh ngang

| Model | Aux loss | Z-loss | Bias adjust | Sequence-level |
|---|---|---|---|---|
| Mixtral | Có (coef 0.001) | Không | Không | Batch |
| Switch | Có | Có | Không | Batch |
| ST-MoE | Có | Có (z-loss) | Không | Batch |
| DeepSeek-V3 | Có (sequence-level only) | Không | Có | Cả hai |
| OLMoE | Có | Có | Không | Batch |
| Qwen3-MoE | Có | Không | Không | Batch |
| GPT-OSS | Có | Không | Không | Batch |
| Jamba | Có | Không | Không | Batch |
| NLLB-MoE | Có (+ expert dropout) | Không | Không | Batch |

## Pitfall

**1. Aux coef quá lớn**: router học balance ưu tiên hơn task. Loss train không giảm.

**2. Aux coef quá nhỏ**: imbalance vẫn xảy ra. Một số expert dead.

**3. Quên multiply by num_experts**: loss scale phụ thuộc E. Default Mixtral nhân E, một số custom implementation quên.

**4. Compute aux loss trên CPU**: `tokens_per_expert` tensor nhỏ nhưng gather qua mọi layer làm CPU bottleneck. Phải compute trên device.

**5. Aux loss ở inference**: bug khi `output_router_logits=True` ở eval. Aux loss không có nghĩa khi `labels=None`, nên check.

**6. Bias adjustment với learning rate sai**: lr quá lớn gây oscillation. Quá nhỏ không converge. DeepSeek paper khuyến nghị `1e-3 / num_experts`.

Chương sau ta đi vào expert capacity và token dropping.
