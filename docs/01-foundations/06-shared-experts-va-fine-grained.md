---
title: Shared experts và fine-grained
---

# Shared experts và fine-grained

Hai innovation gần đây quan trọng nhất của MoE: **shared expert** (luôn dùng cho mọi token) và **fine-grained** (nhiều expert nhỏ thay vì ít expert lớn). Cả hai do DeepSeek đề xuất, sau đó được adopt rộng (OLMoE, Qwen3, một số variant Llama).

## Vấn đề: trade-off giữa specialization và knowledge sharing

Trong Mixtral 8x7B, mỗi expert là một MLP độc lập (~5.6B param). Router chọn 2/8 expert. Quan sát thực nghiệm:

1. **Một số expert chuyên hoá**: expert 3 thường được chọn cho code, expert 5 cho math.
2. **Kiến thức general bị duplicate**: cả 8 expert đều phải biết grammar tiếng Anh, syntax cơ bản, common sense. Vì với token "the", bất kỳ expert nào cũng có thể được chọn.

Vấn đề thứ hai gây lãng phí. 8 expert đều có "knowledge of grammar" sao lưu trong weight, tổng 8 lần redundant.

Ý tưởng giải: tách kiến thức general (shared) khỏi specialized (routed).

## Shared experts: DeepSeek-V2 và V-V3

DeepSeek-V3 có **1 shared expert** luôn active + **256 routed experts** với top-8.

```python
class DeepseekV3MoE(nn.Module):
    """A mixed expert module containing shared experts."""
    def __init__(self, config):
        super().__init__()
        self.experts = DeepseekV3NaiveMoe(config)  # 256 routed experts
        self.gate = DeepseekV3TopkRouter(config)
        if config.n_shared_experts is not None:
            intermediate_size = config.moe_intermediate_size * config.n_shared_experts
            self.shared_experts = DeepseekV3MLP(config=config, intermediate_size=intermediate_size)

    def forward(self, hidden_states):
        residual = hidden_states
        topk_indices, topk_weights = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights)
        if hasattr(self, "shared_experts"):
            hidden_states = hidden_states + self.shared_experts(residual)
        return hidden_states
```

(Lược trích từ `src/transformers/models/deepseek_v3/modeling_deepseek_v3.py`, class `DeepseekV3MoE`.)

Logic:

1. Routed experts xử lý hidden_states với top-k routing.
2. Shared expert xử lý cùng hidden_states (không qua router).
3. Output = routed_output + shared_output.

Mỗi token đi qua:

- 8 routed expert (top-k).
- 1 shared expert (always).
- Total: 9 expert.

Vì sao có lợi:

1. **Shared expert học general knowledge**. Grammar, common sense, sample structure. Không bị duplicate.
2. **Routed experts specialize**. Mỗi expert chuyên một domain hẹp hơn (chỉ cần distinguish, không cần handle universal).
3. **Active params control linh hoạt**. Increase shared size để tăng baseline; tăng routed top-k để tăng specialization.

DeepSeek-V3 config:

```
hidden_size = 7168
moe_intermediate_size = 2048   # mỗi routed expert FFN width
n_routed_experts = 256
num_experts_per_tok = 8        # top-8
n_shared_experts = 1
```

Shared expert có FFN width = `2048 * 1 = 2048`, tức là cùng size như 1 routed expert.

Active params per token:

- Attention: ~4B (MLA share weight)
- Shared expert: ~2.6B (1 expert)
- Routed experts: 8 * 2.6B = ~20.8B
- LM head + embed: ~9.6B
- Total: ~37B

So với 671B total, active 37B = 5.5%.

## Vì sao 1 shared expert thay vì nhiều?

DeepSeek-V2 dùng 2 shared expert. V-V3 giảm xuống 1. Lý do:

1. **1 shared đủ cho general knowledge** với hidden_size lớn (7168).
2. **Routed experts càng nhiều càng specialized**. Tăng n_routed = 256 đem nhiều lợi ích hơn tăng n_shared = 2.
3. **Cost shared expert luôn cộng vào active**. Nhiều shared = nhiều active params cố định.

Một số model khác:

- Qwen3-MoE: không có shared expert (paper experiment thấy không cải thiện).
- OLMoE: không có shared expert.
- Granite-MoE-Shared: có shared expert (theo paper IBM).

Trade-off mơ hồ. Tuỳ data distribution.

## Fine-grained experts

DeepSeek-V2/V3 cũng đi tiên phong **fine-grained**: thay vì ít expert lớn, dùng nhiều expert nhỏ.

So sánh với Mixtral 8x22B:

| Model | num_experts | expert FFN width | total expert params | top-k | active expert params |
|---|---|---|---|---|---|
| Mixtral 8x22B | 8 | ~16K | ~115B | 2 | ~28.8B |
| DeepSeek-V3 (compared) | 256 | 2K | ~106B | 8 | ~3.3B routed (+ 2.6B shared) |

Tổng expert params gần như nhau (~110B). Nhưng:

- Mixtral: 8 expert × 16K FFN. Mỗi expert "lớn" nhưng "ít chuyên".
- DeepSeek-V3: 256 expert × 2K FFN. Mỗi expert "nhỏ" nhưng "rất chuyên".

**Lợi của fine-grained**:

1. **Specialization sharper**. Expert nhỏ học pattern hẹp hơn, sai số ít.
2. **Top-k cao smooth hơn**. 8 trong 256 cho mỗi token combinatorial chọn nhiều hơn (C(256, 8) >> C(8, 2)).
3. **Aux loss easier**. Phân bố đều giữa 256 dễ hơn 8.
4. **EP friendly**. Có thể phân group (n_group=8 × 32 expert/group = 256), placement linh hoạt.

**Hại**:

1. **Router lớn hơn**. `num_experts * hidden_dim` = 256 * 7168 = 1.83M params. Vs Mixtral 8 * 6144 = 49K.
2. **More dispatch overhead**. Mỗi token send/recv qua 8 location thay vì 2. EP all-to-all bandwidth tăng.
3. **Mỗi expert thấy ít token**. Với batch nhỏ, expert có thể "starve".

Xu hướng 2024+ ủng hộ fine-grained. Lý do:

1. PyTorch `grouped_mm` xử lý variable-length tốt.
2. NVLink/InfiniBand bandwidth dư cho dispatch overhead.
3. Quality improvement đáng kể (DeepSeek paper báo cáo +1-2% benchmark).

## OLMoE: fine-grained nhưng không shared

OLMoE-1B-7B (Allen AI) đi theo fine-grained:

```
hidden_size = 2048
intermediate_size = 1024 (mỗi expert)
num_experts = 64
top_k = 8
n_shared_experts = 0
```

64 expert × 1024 FFN size. Không shared. Top-8. Tỉ lệ k/E = 12.5%.

So với DeepSeek-V3: OLMoE nhỏ hơn (7B total) nên không cần shared expert. Fine-grained vẫn tốt ở scale này.

## Implementation: 3D weight tensor

Cả Mixtral, DeepSeek, Qwen3, OLMoE đều dùng **3D weight tensor** cho expert FFN:

```python
class MixtralExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_local_experts
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
```

`gate_up_proj` shape `(E, 2*d_ff, hidden)`. `gate_up_proj[i]` là weight của expert i. Index bằng integer expert ID.

**Tại sao 3D thay vì ModuleList[Linear]?**

1. **Hiệu quả memory**. Single contiguous tensor vs list of separate tensors.
2. **`grouped_mm` requires 3D**. Kernel expect `(E, in, out)` shape.
3. **EP placement đơn giản**. Cắt tensor theo dim 0 (`gate_up_proj[start:end]`) để giữ subset expert trên GPU.

Trade-off: code phức tạp hơn (dùng `index_add_`, `chunk`, etc. thay vì `linear(input)`). Nhưng performance critical.

## So sánh expert sizing strategy

| Model | num_experts | expert size | shared | top-k | Specialty |
|---|---|---|---|---|---|
| Switch Large | 128-2048 | small-medium | no | 1 | High sparsity, encoder-decoder |
| Mixtral 8x7B | 8 | large | no | 2 | Coarse-grained baseline |
| Mixtral 8x22B | 8 | very large | no | 2 | Coarse-grained large |
| DeepSeek-V2 | 160 | medium | 2 shared | 6 | Fine-grained + shared |
| DeepSeek-V3 | 256 | small | 1 shared | 8 | Ultra-fine + shared |
| Qwen3-MoE 30B | 128 | small | no | 8 | Fine-grained no shared |
| Qwen3-MoE 235B | 128 | medium | no | 8 | Fine-grained no shared |
| GPT-OSS-20B | 32 | medium | no | 4 | Medium-grained + MXFP4 |
| GPT-OSS-120B | 128 | small | no | 4 | Fine-grained + MXFP4 |
| OLMoE | 64 | small | no | 8 | Fine-grained open recipe |
| Granite-MoE-Shared | 60 | medium | yes | 8 | Fine + shared, IBM |

Two clusters:

1. **Coarse (k/E > 12%)**: Mixtral, Switch (top-1 of small E). Simpler, less EP burden.
2. **Fine (k/E < 6%)**: DeepSeek, Qwen3, GPT-OSS-120B, OLMoE. Higher quality, harder serving.

## Vai trò của intermediate_size

Một detail: với fine-grained, mỗi expert có FFN width nhỏ. So với dense baseline:

- Dense Llama-3-70B: hidden=8192, intermediate=28672 (3.5x ratio).
- DeepSeek-V3: hidden=7168, moe_intermediate=2048 (0.28x ratio).

Mỗi expert DeepSeek width chỉ 7-8% dense FFN. Vì có 256 expert, total = `256 * 2048 = 524K` capacity, gấp 18x của dense Llama. Đây là "more, smaller" philosophy.

Active = `8 * 2048 = 16K` capacity per token. So với dense Llama 28K (theo router thì gần 60%).

## Pitfall

**1. Quên cộng shared expert output**. Code dễ bug: forward gọi routed nhưng quên `+ shared_output`. Test với num_routed=0 để verify shared standalone work.

**2. Init shared expert quá lớn**: shared dominate, routed không học. Init nhỏ tương tự routed.

**3. Fine-grained nhưng batch nhỏ**: expert starvation. Một số expert nhận 0-1 token, không học. Aux loss giúp, nhưng vẫn cần batch đủ lớn.

**4. 3D weight tensor không init đúng**: `torch.empty` cần init explicit (`init.normal_`). Quên init gây NaN.

**5. EP với shared expert**: shared phải replicate trên mọi GPU (vì mọi token cần nó). Tăng memory mỗi GPU. Phần 4 Chương 2.

Phần 1 kết thúc. Phần 2 đi vào HuggingFace MoE infrastructure.
