---
title: Tensor parallel với MoE
---

# Tensor parallel với MoE

TP shard tensor dimension (column-wise hay row-wise) qua GPU. Đã quen với attention TP (Q/K/V colwise, O rowwise) trong dense Transformer. MoE thêm complexity: expert weight 3D, shard ra sao? Chương này phân tích.

## Recap dense TP

Llama TP plan:

```python
_tp_plan = {
    "model.layers.*.self_attn.q_proj": "colwise",
    "model.layers.*.self_attn.k_proj": "colwise",
    "model.layers.*.self_attn.v_proj": "colwise",
    "model.layers.*.self_attn.o_proj": "rowwise",
    "model.layers.*.mlp.gate_proj": "colwise",
    "model.layers.*.mlp.up_proj": "colwise",
    "model.layers.*.mlp.down_proj": "rowwise",
}
```

Mỗi linear shard theo column (output dim) hoặc row (input dim). Pattern colwise-rowwise-colwise tạo all-reduce ở boundary.

## TP cho MoE: expert weight 3D

MoE expert weight `(E, 2*intermediate, hidden)`. Có 3 dim:

- Dim 0: `E` (num_experts)
- Dim 1: `2*intermediate` (output FFN)
- Dim 2: `hidden` (input)

Shard ra sao:

### Option A: Shard expert dim (dim 0)

```python
gate_up_proj[expert_id]: shard across ranks by expert_id
```

Đây là **expert parallelism (EP)**. Đã thấy Chương 2. Communication: all-to-all.

### Option B: Shard intermediate dim (dim 1)

```python
gate_up_proj[:, intermediate_slice, :]: shard column-wise
```

Tương tự dense TP. Mỗi rank giữ subset của intermediate. Communication: all-reduce sau down projection.

### Option C: Shard hidden dim (dim 2)

```python
gate_up_proj[:, :, hidden_slice]: shard input dim
```

Ít phổ biến. Cần all-reduce input trước linear.

### Option D: 2D shard (combination)

Combo expert + intermediate. Mesh `(EP, TP)`. Mỗi rank giữ subset expert × subset intermediate.

## TP plan typical

Qwen3-MoE example:

```python
class Qwen3MoePreTrainedModel(PreTrainedModel):
    _tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        # MoE-specific
        "model.layers.*.mlp.experts.gate_up_proj": "colwise_experts",
        "model.layers.*.mlp.experts.down_proj": "rowwise_experts",
        "model.layers.*.mlp.gate.weight": "replicate",
    }
```

(Pseudocode, may differ in actual codebase.)

Hai strategy mới:

**`colwise_experts`**: shard `gate_up_proj` along intermediate dim (dim 1). Shape mỗi rank: `(E, 2*intermediate/world_size, hidden)`. Output sau gate có shape `(N*k, 2*intermediate/world_size)`.

**`rowwise_experts`**: shard `down_proj` along intermediate dim (dim 2). Shape mỗi rank: `(E, hidden, intermediate/world_size)`. Input là local part, output cần all-reduce.

Communication boundary: 1 all-reduce mỗi expert pass (sau down_proj). Cộng all-reduce ở attention. Tổng 2 all-reduce per layer.

So với EP (2 all-to-all per layer), TP all-reduce có thể nhanh hơn trên cluster (all-reduce ring-based, optimize tốt).

## TP + EP: 2D mesh

Production cluster lớn (`>= 16 GPU`):

```
World size = 16, mesh (TP=2, EP=8).
Rank layout:
  Group 0 (EP=0, TP={0,1}): GPUs 0, 1 hold experts [0..31] with TP shard
  Group 1 (EP=1, TP={0,1}): GPUs 2, 3 hold experts [32..63]
  ...
  Group 7 (EP=7, TP={0,1}): GPUs 14, 15 hold experts [224..255]
```

Forward:

1. Attention: TP all-reduce within TP group.
2. Router: replicate, all ranks compute.
3. Dispatch: all-to-all across EP groups.
4. Expert: each rank process its share. TP all-reduce within group.
5. Gather: all-to-all reverse.
6. Output: TP all-reduce attention.

Complexity tăng. Cần DTensor framework để manage.

## Router weight: replicate hay shard?

```python
self.gate = nn.Parameter(torch.empty(num_experts, hidden_size))
```

Router weight: `(E, hidden)`. Có thể:

1. **Replicate**: mọi rank giữ full router. Forward redundant nhưng simple. Memory cost: `E * hidden = 256 * 7168 = 1.8M * 4 bytes = 7 MB`. Acceptable.
2. **Shard along E**: rank chỉ giữ router cho local experts. Cần all-gather hoặc compute partial logits + reduce.

HF default: replicate. Cost nhỏ, simplicity worth.

## Shared expert TP

DeepSeek-V3 có 1 shared expert. Shared replicate trên mọi GPU (vì luôn được dùng) nhưng có thể TP shard:

```python
_tp_plan = {
    "model.layers.*.mlp.shared_experts.gate_proj": "colwise",
    "model.layers.*.mlp.shared_experts.up_proj": "colwise",
    "model.layers.*.mlp.shared_experts.down_proj": "rowwise",
}
```

Shared expert là dense MLP, TP shard giống Llama.

## Sequence parallel (SP)

SP shard sequence dim (T) thay vì hidden dim. Hữu ích cho context dài.

```python
_sp_plan = {
    "model.layers.*.input_layernorm": ("scatter", "all_reduce"),
    "model.layers.*.post_attention_layernorm": ("scatter", "all_reduce"),
}
```

SP + TP + EP: 3D mesh. Bandwidth complex hơn nữa.

DeepSeek-V3 train với SP (paper mention). Inference thường không cần.

## FSDP và MoE

FSDP (Fully Sharded Data Parallel): shard parameter theo data-parallel rank. Khác TP:

- TP: shard theo tensor dim, communication intra-tensor.
- FSDP: shard theo param, communication all-gather mỗi forward.

FSDP cho MoE expert: mỗi rank giữ 1/world_size của each expert. All-gather expert weight trước forward, sharded backward.

Cost: FSDP cần all-gather toàn bộ weight mỗi forward. Lớn cho MoE.

Practice:

- FSDP cho dense part (attention, embed, lm_head).
- EP cho expert.
- TP cho intermediate dim.

Combo `_tp_plan + _ep_plan + _fsdp_plan` declare. Framework apply qua DTensor.

## Implementation status trong HF

Tại thời điểm chuỗi bài giảng:

- TP plan: stable, work với `tp_plan="auto"` argument.
- EP plan: develop, experimental.
- SP plan: stable cho attention, develop cho FFN.
- 3D mesh: chưa stable full.

Production thường dùng frameworks bên ngoài: DeepSpeed (cho FSDP+EP), Megatron-LM (cho TP+EP+PP), vLLM (inference). HF tích hợp dần.

## Practical recommendation

| Scenario | Recommended |
|---|---|
| Single GPU (`<= 40GB`) | Quantize (MXFP4 cho GPT-OSS, INT4 cho Mixtral) |
| 2-8 GPU single node | TP `auto` |
| 8-16 GPU single node | TP + small EP |
| Multi-node (32+ GPU) | TP + EP + FSDP via DeepSpeed/Megatron |
| Inference latency | TP only, batch small |
| Inference throughput | EP + continuous batching (vLLM) |

## Pitfall

**1. TP plan với expert weight transposed**: `is_transposed=True` (GPT-OSS) cần TP plan đúng dim. Default colwise/rowwise có thể sai.

**2. Replicate router với 256 expert**: router weight `(256, hidden)` ~1.8M params. Cộng 32 layer = 60M params replicate. Vẫn OK.

**3. FSDP + MoE eager**: expert eager forward không play tốt với FSDP wrap. Phải dùng `grouped_mm` hoặc custom wrap.

**4. EP all-to-all bandwidth fluctuation**: nếu routing skewed, một số rank send/receive nhiều hơn. Latency tail-heavy.

**5. Shared expert + EP**: shared replicate, không shard. Memory mỗi rank: shared + (routed / EP_size).

**6. TP với padding tokens**: nếu sequence không chia hết world_size, pad có thể impact aux loss compute.

Chương sau ta đọc quantization MoE (MXFP4, FP8).
