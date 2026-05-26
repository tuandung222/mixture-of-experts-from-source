---
title: Expert parallelism
---

# Expert parallelism (EP)

EP là kỹ thuật phân phối expert qua nhiều GPU. Mỗi GPU giữ subset expert. Khi forward, token được dispatch đến GPU có expert tương ứng via all-to-all communication. Chương này mô tả EP từ concept đến implementation.

## Vì sao cần EP

Một MoE model có total params lớn (DeepSeek-V3: 671B). Bf16 weight chiếm 1.34 TB. Một H100 có 80 GB. Không thể fit model một GPU.

Hai option:

1. **Pipeline parallel (PP)**: chia model theo layer. Mỗi GPU giữ vài layer. Latency cao vì token đi qua mọi stage.
2. **Expert parallel (EP)**: chia expert theo dim 0 của weight tensor. Mỗi GPU giữ subset expert của mọi layer. Latency tốt hơn vì attention/router replicate.

EP hiệu quả cho MoE vì expert là "wide" dim (256 expert) nên chia tốt. Attention chỉ ~150M params, replicate được.

## EP layout

Ví dụ DeepSeek-V3 với 256 expert, 8 GPU:

```
GPU 0: experts [0, 31]      (32 expert)
GPU 1: experts [32, 63]
GPU 2: experts [64, 95]
GPU 3: experts [96, 127]
GPU 4: experts [128, 159]
GPU 5: experts [160, 191]
GPU 6: experts [192, 223]
GPU 7: experts [224, 255]
```

Mỗi GPU lưu `gate_up_proj[start:end]` và `down_proj[start:end]`. Weight `(32, 2*intermediate, hidden)` thay vì `(256, ...)`. Memory 1/8.

Router weight và attention weight replicate trên mọi GPU.

## Forward flow

1. **Compute hidden_states** (attention output): mọi GPU compute song song (data parallel ở token level).
2. **Compute router_logits**: mọi GPU compute trên local hidden_states.
3. **Determine top-k expert per token**: per-token decision, replicate trên mọi GPU.
4. **All-to-all dispatch**: token được "send" đến GPU có expert mong muốn.
5. **Expert forward**: mỗi GPU process token nhận về với local expert.
6. **All-to-all gather**: kết quả "send back" về GPU gốc.
7. **Weighted sum**: combine top-k output thành final hidden.

Communication overhead: 2x all-to-all per layer (dispatch + gather). Bandwidth requirement cao.

## All-to-all communication

```
Before all-to-all:
  GPU 0 has tokens [t0, t1, t2, t3, t4, t5]
  Routing decisions for each token (top-1 for simplicity):
    t0 -> GPU 1
    t1 -> GPU 0 (self)
    t2 -> GPU 2
    t3 -> GPU 1
    t4 -> GPU 0 (self)
    t5 -> GPU 3

After all-to-all dispatch:
  GPU 0 has: t1, t4 (own tokens) + tokens routed from other GPUs to GPU 0's experts
  GPU 1 has: t0, t3 + ...
  GPU 2 has: t2 + ...
  GPU 3 has: t5 + ...
```

PyTorch `dist.all_to_all` primitive. Implementation:

```python
# Pseudo-code
def expert_parallel_forward(self, hidden_states, top_k_index, top_k_weights):
    local_rank = dist.get_rank()
    world_size = dist.get_world_size()
    experts_per_rank = self.num_experts // world_size

    # Determine destination rank per token-expert pair
    flat_expert_ids = top_k_index.flatten()  # (N*k,)
    dest_ranks = flat_expert_ids // experts_per_rank  # (N*k,)

    # Bucket tokens by dest rank
    tokens_per_rank = [count_tokens_to_rank(r) for r in range(world_size)]

    # All-to-all dispatch
    dispatched_hidden = dist.all_to_all(hidden_states, splits=tokens_per_rank)
    # Now local GPU has tokens for its experts

    # Local expert forward
    output = self.local_experts(dispatched_hidden, local_expert_indices, local_weights)

    # All-to-all gather (reverse)
    gathered_output = dist.all_to_all(output, splits=tokens_per_rank)

    # Reduce: combine top-k for each token
    final = gathered_output.view(N, k, hidden).sum(dim=1)
    return final
```

Chi tiết phức tạp: padding, batching, async overlap. PyTorch DTensor (Distributed Tensor) API handle.

## `RouterParallel` trong HF

HF có abstraction `RouterParallel` (chưa stable, đang develop). Idea:

```python
class RouterParallel(nn.Module):
    """Wrapper around router that handles EP dispatch."""

    def __init__(self, router, mesh):
        self.router = router
        self.mesh = mesh
        self.ep_size = mesh.size("ep")

    def forward(self, hidden_states):
        # Compute logits on all ranks
        router_logits = self.router(hidden_states)
        top_k_index, top_k_weights = ...

        # Mask out experts not on this rank (sentinel)
        local_expert_start = local_rank * (num_experts // world_size)
        local_expert_end = local_expert_start + (num_experts // world_size)
        mask = (top_k_index >= local_expert_start) & (top_k_index < local_expert_end)

        # Tokens with non-local experts → sentinel (zero weight)
        top_k_weights = top_k_weights * mask
        # Adjust index to local
        local_top_k_index = top_k_index - local_expert_start
        local_top_k_index = local_top_k_index.clamp(0, experts_per_rank - 1)

        return local_top_k_index, top_k_weights
```

(Pseudocode dựa trên ý tưởng HF EP implementation.)

Result: mỗi rank xử lý chỉ token thuộc local expert. Token expert thuộc rank khác có weight=0 (skip).

## Sentinel mechanism

```python
sentinel_mask = (expert_ids_g >= self.num_experts).unsqueeze(-1)
expert_ids_g.clamp_(max=self.num_experts - 1)
```

(Đã thấy ở Phần 2 Chương 4.)

Sau RouterParallel, `local_top_k_index` có thể vẫn có expert id "ngoài range local" (do clamping logic). `sentinel_mask` zero out output ở rows này. Cộng dồn output qua all-to-all sẽ hợp lại đúng (token nhận output từ rank có expert).

## All-reduce vs all-to-all

```
Tensor parallel (TP): all-reduce (sum across ranks)
Expert parallel (EP): all-to-all (permute tokens across ranks)
```

Khác biệt:

- All-reduce: bandwidth `O(N * hidden)` total (compute on all, sum).
- All-to-all: bandwidth `O(N * hidden)` total (move tokens, no sum).

Latency cùng order. Implementation kernel khác.

NVLink (intra-node) cho phép all-to-all hiệu quả (300+ GB/s).
InfiniBand (inter-node) chậm hơn (~100-200 GB/s). EP qua nodes có cost cao hơn.

## Hybrid: TP + EP

Production setup thường:

- **TP**: attention + router replicate, expert weight shard column-wise (cho `intermediate_size` dim).
- **EP**: expert ID shard across same/different group.

Ví dụ 32 GPU, mesh `(tp=4, ep=8)`:

- 8 EP groups, mỗi group có 4 GPU.
- Mỗi GPU trong group share 32 expert (256/8) nhưng shard FFN intermediate_size theo TP.
- Group này khác group khác về expert.

Complexity tăng theo cấp số.

## Bandwidth analysis

```
Token N = 4096, hidden = 7168, k = 8, bf16 (2 bytes).
Per token transmission size: hidden * 2 = 14336 bytes.
Total dispatch payload: N * k * 14336 = 4096 * 8 * 14336 = ~470 MB.
Same for gather: 470 MB.
Per forward, per MoE layer: ~940 MB.

DeepSeek-V3 có 58 MoE layer (61 - 3 dense). Total per forward: 940 MB * 58 = ~55 GB.

NVLink (H100): 900 GB/s. Time = 55 / 900 = 0.06s = 60 ms.
InfiniBand (cluster): 100 GB/s. Time = 55 / 100 = 0.55s = 550 ms.
```

Communication is significant fraction of latency. Cluster topology matters.

## Token routing imbalance

EP hiệu quả khi router output balanced. Nếu lệch (1 expert nhận 90% token), GPU đó hottest, còn lại idle.

Aux loss giúp balance, nhưng không guarantee. Practice:

- Train với aux coef đủ (Mixtral 0.001, OLMoE 0.01).
- Eval với capacity factor (Switch) hoặc dropless (Mixtral).
- Monitor expert utilization trong production.

## Sequence-level routing

DeepSeek-V3 có sequence-level aux loss: balance trong mỗi sequence. Lý do EP: communication overhead.

Nếu sequence A có 90% token đi expert 0, sequence B có 90% token đi expert 1, batch-level balanced (50/50). Nhưng:

- Sequence A: tất cả token đi GPU 0. GPU 0 hot, còn lại idle.
- Sequence B: tất cả đi GPU 1.

Sequence-level loss phạt mỗi sequence lệch. Encourage mỗi sequence phân đều, dẫn đến mỗi batch (mỗi micro-step EP) phân đều.

## EP plan trong HF

Tương tự `_tp_plan`:

```python
class DeepseekV3PreTrainedModel(PreTrainedModel):
    _ep_plan = {
        "model.layers.*.mlp.experts.gate_up_proj": "shard_along_experts",
        "model.layers.*.mlp.experts.down_proj": "shard_along_experts",
    }
```

Khai báo expert weight được shard along dim 0 (experts).

Khi `from_pretrained(ep_plan="auto")`, HF apply plan qua DTensor.

(Implementation chưa stable, đang develop trong transformers main branch.)

## Pitfall

**1. Number of GPUs không chia hết num_experts**: ví dụ 256 expert / 6 GPU = 42.67. Phải padding hoặc dùng custom layout.

**2. Imbalanced routing → straggler GPU**: 1 GPU bottleneck cả forward. Aux loss critical.

**3. Communication bottleneck**: InfiniBand cluster slow cho EP. Single-node multi-GPU (NVLink) tốt hơn nhiều.

**4. Sentinel logic sai**: nếu mask không đúng, output sai. Test với EP=1 vs EP=N để verify.

**5. Generate với EP**: decode 1 token, all-to-all cost dominant. Latency cao. Server framework (vLLM) optimize qua continuous batching.

**6. Memory imbalance**: rank giữ shared expert + local routed experts. Shared replicate, routed shard. Tổng memory mỗi rank không đều.

Chương sau ta đọc TP với MoE.
