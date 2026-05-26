---
title: Training recipe MoE
---

# Training recipe MoE

Train MoE từ scratch hoặc fine-tune đòi hỏi recipe khác dense. Chương này tổng hợp practical knowledge: aux loss tune, optimizer, gradient flow, megablocks kernel, fine-tune từ pre-trained.

## Gradient flow MoE

Forward MoE:

```
hidden_states -> router -> top_k_index, top_k_weights
            -> experts (selected) -> output
            -> combine via weights
```

Backward:

1. Loss → output gradient.
2. Output gradient → expert weight gradient (qua expert active).
3. Output gradient × hidden_states → routing_weight gradient (cho expert active).
4. routing_weight gradient → router_logits gradient → router_weight gradient.

Vấn đề: **top-k là argmax-like operation**. Non-differentiable. Token đi expert A có gradient flow expert A, không có signal cho expert B "would have been better".

Solution: dùng routing_weight (softmax prob) làm proxy. `output = sum_i (weight_i * expert_i(x))`. Gradient flow qua weights, dù expert không được chọn (weight = 0) vẫn có gradient ở weight (softmax prob > 0 cho mọi expert).

Subtle: nếu chỉ flow gradient cho top-k expert active (top-k weight > 0), router không học expert khác. Implementation HF flow gradient toàn bộ softmax distribution.

## Aux loss schedule

Aux loss coef: tunable. Best practice:

**1. Warm-up**: aux coef tăng dần từ 0 lên target. Lý do: bắt đầu train, expert chưa specialize. Aux loss force balance quá sớm → expert collapse vào uniform mean. Warm-up cho phép expert phân kỳ trước balance kicks in.

```
Step 0: aux_coef = 0
Step 1000: aux_coef = 0.0001
Step 5000: aux_coef = 0.001 (target Mixtral)
Step 10000+: aux_coef = 0.001
```

**2. Schedule giảm dần**: ngược lại, một số recipe giảm aux coef cuối training. Cho phép quality optimize over balance.

DeepSeek-V3 dùng bias adjustment (không cần aux coef large), nên schedule không matter.

## Optimizer cho MoE

**AdamW**: standard cho LLM. Cho MoE, cùng strategy:

```python
optimizer = AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=0.1,
    betas=(0.9, 0.95),
)
```

**Lưu ý**: expert weight có gradient sparse. Nếu một expert nhận 0 token batch này, gradient = 0. Adam momentum vẫn update (decay), gây drift.

Solution:

1. **Sparse-aware Adam**: skip update khi grad=0. Implementation custom.
2. **Aux loss đủ** để mọi expert được chạm trong mỗi batch.
3. **Ignore**: practice phổ biến, drift nhỏ acceptable.

Mixtral, OLMoE, Qwen3 đều dùng standard AdamW. Drift không significant.

## Learning rate cho expert

Một số paper đề xuất router lr khác expert lr:

```
router_lr = 0.5 * base_lr      # router cần stable
expert_lr = 1.0 * base_lr      # expert update normal
```

Lý do: router weight quyết định route, sensitive. Expert weight nhiều, gradient noisy. Smaller lr cho router stable hơn.

Mixtral paper không mention. DeepSeek-V3 dùng cùng lr (theo paper).

Practice: thường không cần. Optional optimization.

## Megablocks kernel

Megablocks (Stanford/MosaicML, 2023) là kernel CUDA optimize cho dropless MoE. Pattern: block-sparse matmul.

```
Mixtral with batch 4096 token, top-2:
  Total assignment: 4096 × 2 = 8192 token-expert pairs.
  Distribute across 8 expert.

Megablocks block-sparse:
  Each expert: ~1024 token (if balanced).
  Pad to block size 128: each expert has 8 blocks of 128 tokens.
  Block matmul: 8 expert × 8 blocks × matmul(128, hidden, intermediate).
  Hardware-friendly.
```

PyTorch `torch._grouped_mm` essentially megablocks API. Stanford released megablocks library trước; PyTorch absorb 2024.

Speed: 2-5x vs naive for-loop.

Phần 2 Chương 4 đã đi sâu `grouped_mm`.

## Fine-tune Mixtral

Most popular fine-tune target. Recipe:

```python
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    Trainer, TrainingArguments,
)
from peft import LoraConfig, get_peft_model

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_router_logits=True,    # cần cho aux loss
)

# LoRA target: attention + router (NOT expert!)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate",   # router, NOT experts
    ],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="./mixtral_lora",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=3,
    bf16=True,
    save_strategy="epoch",
)

trainer = Trainer(model=model, args=training_args, train_dataset=dataset)
trainer.train()
```

**Note**:

1. **Không LoRA expert**: expert có 3D tensor, LoRA chuẩn không apply trực tiếp. Cần custom impl hoặc skip.
2. **LoRA router**: làm router adapt theo task.
3. **Aux loss enable**: `output_router_logits=True`. Trainer apply nếu config có.
4. **Batch nhỏ** (per_device=2): Mixtral 47B + LoRA cần memory. Gradient accumulation tăng effective batch.

## Common bugs khi fine-tune

**1. `output_router_logits=False`**: aux loss = 0. Expert không balance, fine-tune bị router collapse sau vài hundred steps. Quality drop drastic.

Fix: set `output_router_logits=True` trong config hoặc forward kwargs.

**2. LoRA hit expert nhưng dimension mismatch**: expert weight 3D, LoRA expect 2D. Crash.

Fix: exclude expert from LoRA target. Hoặc dùng LoRA-MoE custom (apply per-expert LoRA).

**3. Aux loss exploding**: aux coef quá lớn cho fine-tune. Default 0.001 cho pre-train. Fine-tune nên thấp hơn (0.0001) hoặc disable.

**4. Gradient checkpointing crash với MoE**: standard `gradient_checkpointing_enable()` không play tốt với expert dispatch. Có thể cần custom.

**5. Memory spike batch 1 prefill**: MoE prefill compute mọi expert qua batch token. Memory peak. Reduce sequence length hoặc prefill chunked.

## Continue pre-training

Add data, continue from checkpoint:

```python
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-v0.1")
# Note: v0.1 (base), not instruct

# Continue pre-training
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        learning_rate=1e-5,   # Lower than initial pretrain
        warmup_steps=1000,
        ...
    ),
    train_dataset=domain_data,
)
trainer.train()
```

**Note**:

1. **Lower lr**: model đã hội tụ, lr cao destroy weight. 10-100x nhỏ hơn initial pretrain.
2. **Warm-up**: even shorter warm-up.
3. **Aux coef**: giữ Mixtral original (0.001).
4. **Data**: domain-specific (code, math, ...). Aux loss giúp expert specialize theo domain.

## Train from scratch

Hardest setup. Requirement:

- GPU cluster 100+ H100.
- Data 5-15T tokens.
- Trainer framework (Megatron-LM, DeepSpeed, FSDP).
- Aux schedule tuned.
- Router init careful.

Single individual không train from scratch một MoE 30B+. Companies (Mistral, DeepSeek, Microsoft) do.

Recipe public:

- OLMoE: open recipe + code (AI2).
- Mixtral: paper details (Mistral).
- DeepSeek-V3: paper details.

## Pitfall summary

**1. Disable aux loss khi fine-tune ngắn**: model đã balance từ pretrain, aux loss force re-balance gây regression. Cân nhắc.

**2. LoRA toàn bộ model bao gồm expert**: implementation phức tạp. Default LoRA target = attention + router.

**3. Token routing imbalance trong fine-tune**: nếu data domain-specific (chỉ code), expert nào "chuyên code" được chọn quá nhiều. Aux loss vẫn cần.

**4. Quên `output_router_logits` lúc eval**: aux loss = 0 ở eval log. OK (không bug), nhưng metric không match training.

**5. Tied weights và LoRA**: lm_head tied với embed. Apply LoRA cho embed → lm_head cũng adapt. Note carefully.

**6. Cache với LoRA train**: KV cache trong training thường không dùng (forward full sequence). MoE tương tự, không tận dụng cache.

Phần 4 kết thúc. Sang Phần 5: design comparison + decision guide.
