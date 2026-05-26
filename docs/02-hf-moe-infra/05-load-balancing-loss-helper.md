---
title: load_balancing_loss_func helper
---

# `load_balancing_loss_func` helper

Hàm này không nằm trong `integrations/moe.py` (chứa runtime forward). Nó nằm trong từng `modeling_*moe*.py` (Mixtral, Qwen3-MoE, OLMoE, ...) nhưng implementation nhân bản gần như identical. Chương này phân tích để tránh phải đọc lại ở Phần 3.

## Vị trí

Mỗi model MoE có một bản:

```
src/transformers/models/mixtral/modeling_mixtral.py: load_balancing_loss_func
src/transformers/models/qwen3_moe/modeling_qwen3_moe.py: load_balancing_loss_func
src/transformers/models/olmoe/modeling_olmoe.py: load_balancing_loss_func
...
```

Tại sao không có một bản share? Lý do lịch sử: Mixtral đi đầu, các model sau copy. HF philosophy "one-model one-file" giữ duplicate này. Modular files (Qwen3 từ Mixtral) giúp sync.

## Signature

```python
def load_balancing_loss_func(
    gate_logits: torch.Tensor | tuple[torch.Tensor] | None,
    num_experts: int | None = None,
    top_k=2,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor | int:
```

**Input**:

- `gate_logits`: tuple, mỗi element là gate logits của một layer. Shape mỗi `(B*S, num_experts)`. Tuple length = số layer MoE.
- `num_experts`: scalar.
- `top_k`: scalar (Mixtral 2, DeepSeek 8, ...).
- `attention_mask`: mask cho padding token, shape `(B, S)` hoặc None.

**Output**: scalar loss tensor, hoặc `0` (int) nếu input None.

## Implementation

```python
def load_balancing_loss_func(gate_logits, num_experts, top_k, attention_mask):
    if gate_logits is None or not isinstance(gate_logits, tuple):
        return 0

    compute_device = gate_logits[0].device
    concatenated_gate_logits = torch.cat([layer.to(compute_device) for layer in gate_logits], dim=0)

    routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
    _, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)

    if attention_mask is None:
        tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
        router_prob_per_expert = torch.mean(routing_weights, dim=0)
    else:
        # Padding aware computation
        batch_size, sequence_length = attention_mask.shape
        num_hidden_layers = concatenated_gate_logits.shape[0] // (batch_size * sequence_length)

        expert_attention_mask = (
            attention_mask[None, :, :, None, None]
            .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
            .reshape(-1, top_k, num_experts)
            .to(compute_device)
        )
        tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(
            expert_attention_mask, dim=0
        )
        router_per_expert_attention_mask = (
            attention_mask[None, :, :, None]
            .expand((num_hidden_layers, batch_size, sequence_length, num_experts))
            .reshape(-1, num_experts)
            .to(compute_device)
        )
        router_prob_per_expert = torch.sum(routing_weights * router_per_expert_attention_mask, dim=0) / torch.sum(
            router_per_expert_attention_mask, dim=0
        )

    overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
    return overall_loss * num_experts
```

(Đầy đủ từ `src/transformers/models/mixtral/modeling_mixtral.py`, function `load_balancing_loss_func`.)

Đi từng bước:

**Step 1: Early return**

```python
if gate_logits is None or not isinstance(gate_logits, tuple):
    return 0
```

Khi model forward không có `output_router_logits=True`, gate_logits = None. Loss = 0 (no aux loss). Int 0 ok cho `loss += router_aux_loss_coef * 0` (no NaN).

**Step 2: Concatenate layers**

```python
concatenated_gate_logits = torch.cat([layer.to(compute_device) for layer in gate_logits], dim=0)
```

Mỗi layer gate logits là `(B*S, E)`. Cat dim=0 cho `(L*B*S, E)`. Coi mọi token mọi layer là sample độc lập.

`compute_device`: device của layer đầu tiên. Mỗi layer có thể trên device khác (TP/PP), move về một device để compute.

**Step 3: Softmax**

```python
routing_weights = torch.nn.functional.softmax(concatenated_gate_logits, dim=-1)
```

Note: dùng default dtype (bf16 nếu model bf16). Có thể cast fp32 explicit nếu cần stability, nhưng aux loss có coef nhỏ, không quan trọng.

**Step 4: Top-k**

```python
_, selected_experts = torch.topk(routing_weights, top_k, dim=-1)
```

Output shape `(L*B*S, top_k)`, int indices.

**Step 5: One-hot mask**

```python
expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
```

Shape `(L*B*S, top_k, num_experts)`. Mỗi token-position-k có 1 ở expert id selected, 0 elsewhere.

**Step 6a: No attention mask**

```python
tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
# Shape: (top_k, num_experts), mỗi value là fraction of tokens selecting that expert at that k position

router_prob_per_expert = torch.mean(routing_weights, dim=0)
# Shape: (num_experts), mean probability
```

**Step 6b: With attention mask** (more complex):

Construct `expert_attention_mask` shape `(L*B*S, top_k, num_experts)`:

```python
expert_attention_mask = (
    attention_mask[None, :, :, None, None]
    .expand((num_hidden_layers, batch_size, sequence_length, top_k, num_experts))
    .reshape(-1, top_k, num_experts)
)
```

Expand attention mask qua layer/k/expert dim. Reshape thành cùng shape với `expert_mask`. Token padding → 0 ở mọi position trong expert mask.

Compute với mask:

```python
tokens_per_expert = torch.sum(expert_mask.float() * expert_attention_mask, dim=0) / torch.sum(expert_attention_mask, dim=0)
# Mean qua non-padding tokens only
```

Tương tự cho `router_prob_per_expert`.

**Step 7: Final loss**

```python
overall_loss = torch.sum(tokens_per_expert * router_prob_per_expert.unsqueeze(0))
return overall_loss * num_experts
```

`tokens_per_expert` shape `(top_k, num_experts)`. `router_prob_per_expert.unsqueeze(0)` shape `(1, num_experts)`. Broadcast multiply `(top_k, num_experts)`. Sum tất cả.

Multiply by `num_experts` để normalize: uniform distribution cho loss = 1.

**Verification**:

```
Uniform case:
  tokens_per_expert[k][e] = 1/num_experts for all e
  router_prob_per_expert[e] = 1/num_experts for all e
  product[k][e] = 1/(num_experts^2)
  sum over (top_k, num_experts) = top_k * num_experts * 1/(num_experts^2) = top_k / num_experts
  * num_experts = top_k

Hmm, this gives top_k, not 1.
```

Wait, the formula in Switch paper eq 4 gives loss = 1 at uniform. Implementation here may differ slightly. Reading more carefully:

```python
tokens_per_expert = torch.mean(expert_mask.float(), dim=0)
# shape (top_k, num_experts)
# Each value = fraction of tokens routed to expert e at top-k position k
# Sum over experts at one k position = 1
```

OK so `tokens_per_expert.sum(dim=-1) = 1` (per k position).

```python
router_prob_per_expert = torch.mean(routing_weights, dim=0)
# shape (num_experts)
# Sum over experts = 1 (softmax sum)
```

`router_prob_per_expert.sum() = 1`.

Product `tokens_per_expert[k][e] * router_prob_per_expert[e]`:

```
sum_e (tokens_per_expert[k][e] * router_prob_per_expert[e])
```

is dot product of two vectors. By Cauchy-Schwarz, minimum when uniform: each = `(1/E) * (1/E) = 1/E²`. Sum over E experts: `1/E`. Then sum over top_k positions (since `tokens_per_expert` has shape `(k, E)`):

```
total = top_k * (1/E)
```

Multiply by `num_experts`:

```
loss = top_k * (1/E) * E = top_k
```

So uniform loss = top_k. For Mixtral top_k=2, uniform aux_loss = 2. Mixtral applies coef 0.001: `total_loss = ce_loss + 0.001 * 2 = ce_loss + 0.002`. Negligible.

When imbalanced, both `tokens_per_expert` và `router_prob_per_expert` deviate from uniform, dot product increases. So loss > top_k. Gradient pushes back to uniform.

(Đây là chi tiết nhỏ HF không document rõ. Trong Switch paper, scale có thể khác.)

## Khi nào aux loss được tính

Trong forward của top-level model (như `MixtralForCausalLM`):

```python
def forward(self, input_ids, ..., output_router_logits=None, ...):
    if output_router_logits is None:
        output_router_logits = self.config.output_router_logits

    outputs = self.model(input_ids, ..., output_router_logits=output_router_logits, ...)

    logits = self.lm_head(outputs.last_hidden_state)
    loss = None
    aux_loss = None

    if labels is not None:
        loss = self.loss_function(logits, labels, ...)
        if output_router_logits:
            aux_loss = load_balancing_loss_func(
                outputs.router_logits, self.config.num_local_experts,
                self.config.num_experts_per_tok, attention_mask,
            )
            loss += self.config.router_aux_loss_coef * aux_loss.to(loss.device)

    return MoeCausalLMOutputWithPast(
        loss=loss, aux_loss=aux_loss, logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        router_logits=outputs.router_logits,
    )
```

(Lược trích pattern chung. `src/transformers/models/mixtral/modeling_mixtral.py`, class `MixtralForCausalLM.forward`.)

Khi nào `output_router_logits = True`?

1. **Training**: set qua `config.output_router_logits = True` lúc init, hoặc pass argument trong forward.
2. **Eval**: thường False (không cần aux loss, save memory).

Khi training với HF Trainer, `output_router_logits` được force True nếu aux loss enable.

## Output dataclass: `MoeCausalLMOutputWithPast`

```python
@dataclass
class MoeCausalLMOutputWithPast(ModelOutput):
    loss: torch.FloatTensor | None = None
    aux_loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor = None
    past_key_values: Cache | None = None
    hidden_states: tuple[torch.FloatTensor] | None = None
    attentions: tuple[torch.FloatTensor] | None = None
    router_logits: tuple[torch.FloatTensor] | None = None
```

(`src/transformers/modeling_outputs.py`.)

Khác `CausalLMOutputWithPast` (dense model): có thêm `aux_loss` và `router_logits`. Trainer/eval framework có thể inspect `aux_loss` riêng cho metric.

## Coef per model

| Model | router_aux_loss_coef | Note |
|---|---|---|
| Mixtral | 0.001 | Standard |
| Switch | 0.01 (aux) + 0.001 (z) | Aux + z-loss |
| DeepSeek-V3 | 0.0001 (sequence-only) | Mostly bias-based |
| Qwen3-MoE | 0.001 | Mixtral-style |
| OLMoE | 0.01 (aux) + 0.01 (z) | Both losses |
| GPT-OSS | 0.001 | Mixtral-style |
| Jamba | 0.001 | Mixtral-style |
| NLLB-MoE | 0.01 | + expert dropout |
| PhiMoE | 0.001 | Mixtral-style |

Coef nhỏ (0.0001 - 0.01). Loss aux không dominate task loss.

## Pitfall

**1. Quên `attention_mask`**: padding token đi vào aux loss, inflate stats. Forward phải pass mask.

**2. Aux loss compute trên CPU**: nếu gate_logits chuyển CPU (debug), aux loss compute chậm. Phải giữ device.

**3. Aux loss với batch=1**: với 1 token, distribution không meaningful. Aux loss noisy. Eval với batch lớn.

**4. Aux loss khi inference**: bug khi `output_router_logits=True` ở eval mà không có labels. Aux loss compute nhưng không dùng. Memory waste, không sai.

**5. Aux loss với group routing**: DeepSeek-V3 routing 2 tầng (group → expert). Aux loss standard không capture group-level. DeepSeek có aux loss riêng.

Phần 2 kết thúc. Phần 3 sang model walkthroughs, bắt đầu với Mixtral.
