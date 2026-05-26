---
title: Cheatsheet
---

# Cheatsheet

Snippet copy-paste-ready cho MoE. Không giải thích dài.

## Class structure mẫu

```python
# Standard 2024+ MoE pattern
@use_experts_implementation
class XExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, top_k_index, top_k_weights):
        # Eager fallback
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

## Router mẫu (Mixtral style)

```python
class XTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.weight = nn.Parameter(torch.empty(self.num_experts, config.hidden_size))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        return router_logits, router_top_value, router_indices
```

## SparseMoeBlock

```python
class XSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = XTopKRouter(config)
        self.experts = XExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states, router_logits
```

## Load model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    output_router_logits=True,    # cần cho aux loss khi train
)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
```

## TP plan

```python
class XPreTrainedModel(PreTrainedModel):
    _tp_plan = {
        "model.layers.*.self_attn.q_proj": "colwise",
        "model.layers.*.self_attn.k_proj": "colwise",
        "model.layers.*.self_attn.v_proj": "colwise",
        "model.layers.*.self_attn.o_proj": "rowwise",
        "model.layers.*.mlp.experts.gate_up_proj": "colwise_experts",
        "model.layers.*.mlp.experts.down_proj": "rowwise_experts",
        "model.layers.*.mlp.gate.weight": "replicate",
    }
```

## Load với TP

```python
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mixtral-8x7B-Instruct-v0.1",
    torch_dtype=torch.bfloat16,
    tp_plan="auto",
)
# Cần torchrun --nproc-per-node=4 python script.py
```

## Aux loss compute

```python
from transformers.models.mixtral.modeling_mixtral import load_balancing_loss_func

aux_loss = load_balancing_loss_func(
    gate_logits=outputs.router_logits,
    num_experts=config.num_local_experts,
    top_k=config.num_experts_per_tok,
    attention_mask=attention_mask,
)

total_loss = ce_loss + 0.001 * aux_loss
```

## Fine-tune LoRA Mixtral

```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, lora_config)
```

## Quantize MXFP4 (GPT-OSS only)

```python
from transformers import AutoModelForCausalLM, Mxfp4Config

quant_config = Mxfp4Config(
    modules_to_not_convert=["lm_head", "gate"],
)

model = AutoModelForCausalLM.from_pretrained(
    "openai/gpt-oss-120b",
    quantization_config=quant_config,
    device_map="auto",
)
```

## Inference với vLLM

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    tensor_parallel_size=2,
    max_num_batched_tokens=8192,
    quantization="awq",  # optional
)

prompts = ["Hello, who are you?"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=200)
outputs = llm.generate(prompts, sampling_params)
```

## Verify config

```python
# Check MoE-specific config
print(f"num_experts: {model.config.num_local_experts}")
print(f"top_k: {model.config.num_experts_per_tok}")
print(f"aux_coef: {model.config.router_aux_loss_coef}")
print(f"output_router_logits: {model.config.output_router_logits}")
print(f"experts_implementation: {model.config._experts_implementation}")
```

## Lookup table: model → key configs

```
Mixtral 8x7B:
  num_local_experts = 8, num_experts_per_tok = 2
  router_aux_loss_coef = 0.001
  hidden = 4096, intermediate = 14336

DeepSeek-V3:
  n_routed_experts = 256, n_shared_experts = 1
  num_experts_per_tok = 8
  n_group = 8, topk_group = 4
  hidden = 7168, moe_intermediate = 2048

Qwen3-30B-A3B:
  num_experts = 128, num_experts_per_tok = 8
  hidden = 2048, moe_intermediate = 768

GPT-OSS-120B:
  num_local_experts = 128, num_experts_per_tok = 4
  hidden = 2880, intermediate = 2880
  attention_bias = True
  quantization_config = Mxfp4Config(...)
```

## Pitfall reminders

```
1. Quên output_router_logits=True ở train → aux loss = 0 → expert collapse.
2. Aux loss coef quá lớn → task loss conflict.
3. EP với num_GPUs không chia hết num_experts → error.
4. LoRA expert.gate_up_proj → dimension mismatch (3D).
5. Quantize router → route collapse. Always exclude.
6. Single GPU batch=1 MoE → tệ hơn dense. Cần batch ≥ 4.
```

## Reference table: paradigm match

```
Cần baseline reference: Mixtral
Cần encoder-decoder: Switch / NLLB-MoE
Cần SOTA quality: DeepSeek-V3
Cần modern infra: Qwen3-MoE / OLMoE
Cần production quant: GPT-OSS
Cần open recipe: OLMoE
Cần MoA + MoE: JetMoE
Cần Mamba + MoE: Jamba
Cần translation: NLLB-MoE
Cần small-scale: PhiMoE
```
