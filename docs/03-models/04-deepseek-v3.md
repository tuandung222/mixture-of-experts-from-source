---
title: DeepSeek-V3
---

# DeepSeek-V3

DeepSeek-V3 (DeepSeek-AI, December 2024) là một trong những model MoE open-weight mạnh nhất 2024. 671B total params, 37B active, sparsity 5.5%. Đại diện cho hướng "fine-grained + shared + aux-free balancing". Chương này đi sâu thiết kế đặc thù.

## Context

- **Tác giả**: DeepSeek-AI (Trung Quốc).
- **Release**: December 2024 (V3); January 2025 (V3-Base, V3 Chat).
- **Paper**: "DeepSeek-V3 Technical Report" (DeepSeek-AI, 2024).
- **License**: DeepSeek License (commercial OK).
- **Variants**: DeepSeek-V3, DeepSeek-V3-Base, DeepSeek-V3.1 (incremental).

## Config key

```python
class DeepseekV3Config:
    hidden_size = 7168
    intermediate_size = 18432            # for dense layers
    moe_intermediate_size = 2048         # mỗi routed expert
    n_routed_experts = 256
    num_experts_per_tok = 8              # top-k routed
    n_shared_experts = 1
    n_group = 8                          # group routing
    topk_group = 4                       # số group được chọn
    norm_topk_prob = True
    routed_scaling_factor = 2.5           # scale routing weight
    num_hidden_layers = 61
    first_k_dense_replace = 3             # 3 layer đầu là dense, sau đó MoE
    num_attention_heads = 128
    num_key_value_heads = 128             # MHA + MLA
    q_lora_rank = 1536                    # MLA: compress Q
    kv_lora_rank = 512                    # MLA: compress KV
    qk_rope_head_dim = 64
    qk_nope_head_dim = 128
    v_head_dim = 128
    vocab_size = 129280
```

Active params per token:

- Attention (MLA): ~4.5B
- 8 routed experts × ~3.3B (with sparsity): ~26B
- 1 shared expert: ~2.6B
- Embedding + lm_head + norm: ~3.9B
- Total: ~37B active

Total: 671B.

## Cấu trúc

```
modeling_deepseek_v3.py (725 dòng)
├── DeepseekV3MLP                  # Dense MLP (cho 3 layer đầu và shared expert)
├── DeepseekV3TopkRouter           # Router: linear + sigmoid + bias correction
├── DeepseekV3NaiveMoe             # 3D weight, @use_experts_implementation
├── DeepseekV3MoE                  # SparseMoeBlock với shared expert + group routing
├── DeepseekV3Attention            # MLA (Multi-head Latent Attention)
├── DeepseekV3DecoderLayer
├── DeepseekV3PreTrainedModel
├── DeepseekV3Model
└── DeepseekV3ForCausalLM
```

## `DeepseekV3TopkRouter`

```python
class DeepseekV3TopkRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_routed_experts = config.n_routed_experts

        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, config.hidden_size)))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts))

    def forward(self, hidden_states):
        hidden_states = hidden_states.view(-1, self.config.hidden_size)
        router_logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        return router_logits
```

(`src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`, class `DeepseekV3TopkRouter`.)

Khác Mixtral:

1. **Bias correction là `register_buffer`**, không phải `Parameter`. Tức là không được tối ưu qua autograd. Update qua callback ngoài training step (xem Phần 1 Chương 4).
2. **Linear ở fp32 explicit** (cast input + weight). Conservative cho 256 expert.
3. **Không return scores/indices**. Class chỉ output logits. Logic top-k, sigmoid, group routing nằm ở `DeepseekV3MoE.route_tokens_to_experts`.

## `DeepseekV3MoE`

```python
class DeepseekV3MoE(nn.Module):
    """A mixed expert module containing shared experts."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.experts = DeepseekV3NaiveMoe(config)
        self.gate = DeepseekV3TopkRouter(config)
        self.shared_experts = DeepseekV3MLP(
            config=config, intermediate_size=config.moe_intermediate_size * config.n_shared_experts
        )
        self.n_routed_experts = config.n_routed_experts
        self.n_group = config.n_group
        self.topk_group = config.topk_group
        self.norm_topk_prob = config.norm_topk_prob
        self.routed_scaling_factor = config.routed_scaling_factor
        self.top_k = config.num_experts_per_tok

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
            topk_weights /= denominator
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states
```

(class `DeepseekV3MoE`.)

Đây là **chỗ rất quan trọng**. Đi từng dòng:

### `route_tokens_to_experts`

**Bước 1: Sigmoid (không softmax)**

```python
router_logits = router_logits.sigmoid()
```

Khác Mixtral (softmax). Mỗi expert score độc lập `[0, 1]`, không cạnh tranh sum=1.

**Bước 2: Bias correction**

```python
router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
```

Cộng bias `(num_experts,)` vào score. Bias này update qua callback (không gradient). Khi expert underutilized, bias tăng → score tăng → expert được chọn nhiều hơn ở step sau. Aux-free balancing.

Lưu ý: chỉ dùng bias **cho choice** (`for_choice` suffix). Khi combine output, dùng `router_logits` gốc (không bias). Bias không méo output.

**Bước 3: Group scoring**

```python
group_scores = (
    router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
    .topk(2, dim=-1)[0]
    .sum(dim=-1)
)
```

Reshape `(N, 256)` → `(N, 8, 32)` (8 group, 32 expert mỗi group). Trong mỗi group, lấy top-2 score, sum. Result `(N, 8)`: tổng top-2 trong mỗi group.

Hợp lý: group có nhiều expert mạnh thì score cao.

**Bước 4: Pick top groups**

```python
group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
group_mask = torch.zeros_like(group_scores)
group_mask.scatter_(1, group_idx, 1)
```

Chọn `topk_group = 4` group có score cao nhất. `group_mask` shape `(N, 8)` với 1 ở group được chọn.

**Bước 5: Expand group mask to expert mask**

```python
score_mask = (
    group_mask.unsqueeze(-1)
    .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
    .reshape(-1, self.n_routed_experts)
)
```

Mask `(N, 8)` → `(N, 8, 32)` → `(N, 256)`. Mỗi expert trong group selected → 1; expert trong group bị loại → 0.

**Bước 6: Mask scores**

```python
scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), float("-inf"))
```

Expert ngoài group selected có score = `-inf` (sẽ không bao giờ được chọn).

**Bước 7: Final top-k**

```python
topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
topk_weights = router_logits.gather(1, topk_indices)
```

Chọn top-8 từ scores. **Quan trọng**: `topk_weights` được gather từ `router_logits` (sigmoid output, không bias). Vì bias chỉ để balance choice, không phải để weight output.

**Bước 8: Normalize**

```python
if self.norm_topk_prob:
    denominator = topk_weights.sum(dim=-1, keepdim=True) + 1e-20
    topk_weights /= denominator
```

Renormalize sum to 1 trong top-k. `1e-20` epsilon tránh div by 0.

**Bước 9: Scale**

```python
topk_weights = topk_weights * self.routed_scaling_factor
```

Scale = 2.5. Khác Mixtral (không scale). Vì sao? Sigmoid output thường nhỏ hơn softmax (mỗi expert độc lập, không sum=1). Scale up để combine output có magnitude đúng.

### `forward`

```python
def forward(self, hidden_states):
    residuals = hidden_states
    router_logits = self.gate(hidden_states)
    topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
    hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
    hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
    hidden_states = hidden_states + self.shared_experts(residuals)
    return hidden_states
```

Flow:

1. Compute router_logits.
2. Route through group → top-k.
3. Dispatch through `self.experts` (routed).
4. **Add shared expert output** (dùng `residuals` = original hidden_states).

Shared expert luôn được apply, song song với routed.

## `DeepseekV3NaiveMoe`

```python
@use_experts_implementation
class DeepseekV3NaiveMoe(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager loop identical to MixtralExperts
        ...
```

Identical với MixtralExperts về weight layout và forward logic. Khác:

1. **Tên `NaiveMoe`** thay vì `Experts`. Naming choice của DeepSeek team.
2. `config.num_local_experts` = `n_routed_experts` (DeepSeek-V3 alias). Trong config check `num_local_experts` cho compat.
3. `intermediate_dim = moe_intermediate_size = 2048` (nhỏ vì fine-grained).

## MLA attention (briefly)

DeepSeek-V3 dùng **Multi-head Latent Attention** (MLA): compress KV qua low-rank projection, tiết kiệm cache.

```python
class DeepseekV3Attention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_a_proj = nn.Linear(hidden_size, q_lora_rank, bias=False)
        self.q_a_layernorm = ...
        self.q_b_proj = nn.Linear(q_lora_rank, num_heads * (qk_nope_head_dim + qk_rope_head_dim), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(hidden_size, kv_lora_rank + qk_rope_head_dim, bias=False)
        self.kv_a_layernorm = ...
        self.kv_b_proj = nn.Linear(kv_lora_rank, num_heads * (qk_nope_head_dim + v_head_dim), bias=False)
        ...
```

Hai bước projection cho Q, K, V qua latent space (rank 512 cho KV). Cache lưu `kv_a` (compressed) thay vì full `K, V`. Tiết kiệm 5-10x KV cache memory so với MHA standard.

MLA không liên quan trực tiếp MoE, không đi sâu ở chuỗi này. Tham khảo DeepSeek-V2 paper.

## `DeepseekV3DecoderLayer`

```python
class DeepseekV3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = DeepseekV3Attention(config, layer_idx)
        # Layer đầu tiên là dense, sau đó MoE
        if layer_idx < config.first_k_dense_replace:
            self.mlp = DeepseekV3MLP(config)
        else:
            self.mlp = DeepseekV3MoE(config)
        ...
```

3 layer đầu (`first_k_dense_replace = 3`) là dense MLP. 58 layer sau là MoE.

Lý do? DeepSeek paper: layer đầu xử lý input pattern thấp, sparse routing chưa có nhiều thông tin. Dense layer ổn định hơn.

## Aux loss DeepSeek

DeepSeek-V3 dùng:

1. **Bias adjustment** (không phải gradient): chính. Coef không có.
2. **Sequence-level aux loss**: phụ. Coef = 0.0001.

Sequence-level: tính aux loss qua mỗi sequence riêng, không qua batch. Encourage mỗi sequence cũng balance.

## So sánh nhanh với Mixtral

| Aspect | Mixtral | DeepSeek-V3 |
|---|---|---|
| Router norm | Softmax | Sigmoid |
| Top-k | 2 | 8 |
| Num experts | 8 | 256 |
| Group routing | Không | Có (8 group, top 4) |
| Shared expert | Không | Có (1) |
| Aux loss | Standard (coef 0.001) | Sequence-level (0.0001) + bias adjust |
| Routing scaling | Không | 2.5x |
| Layer pattern | Mọi layer MoE | 3 đầu dense, còn lại MoE |
| Attention | GQA | MLA |
| Active/Total | 28% | 5.5% |

## Pitfall

**1. Bias là buffer, không phải parameter**: không qua optimizer. Cần custom callback update. Implement sai → bias không thay đổi → balance fail.

**2. Sigmoid score không sum=1**: nếu user expect softmax behavior, output nhiễu.

**3. Group routing với batch=1**: chỉ 1 token, group score noisy. Inference user có thể thấy expert distribution không như ý.

**4. `routed_scaling_factor` quên scale**: output magnitude sai 2.5x.

**5. Shared expert phải replicate trên mọi GPU**: với EP, shared luôn cần present. Tăng memory mỗi GPU.

**6. MLA và KV cache**: cache layout khác hẳn MHA. Code generation phải support.

Chương sau ta đọc Qwen3-MoE.
