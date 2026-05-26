---
title: Mixtral (baseline)
---

# Mixtral

Mixtral 8x7B (Mistral AI, tháng 12/2023) là model MoE LLM open-weight đầu tiên có scale production. Apache 2.0, weight công khai. Trở thành baseline mà mọi model MoE sau so sánh. Chương này đọc kỹ vì đây là reference cho 9 model còn lại.

## Context

- **Tác giả**: Mistral AI.
- **Release**: December 2023 (8x7B), April 2024 (8x22B).
- **Paper**: "Mixtral of Experts" (Jiang et al., 2024).
- **License**: Apache 2.0.
- **Variant**: 8x7B (Mixtral-8x7B-v0.1), 8x22B (Mixtral-8x22B-v0.1).

## Config key

Mixtral 8x7B (`mistralai/Mixtral-8x7B-v0.1`):

```python
class MixtralConfig:
    hidden_size = 4096
    intermediate_size = 14336        # ratio 3.5x (SwiGLU)
    num_hidden_layers = 32
    num_attention_heads = 32
    num_key_value_heads = 8          # GQA 4:1
    num_local_experts = 8            # E
    num_experts_per_tok = 2          # k
    router_aux_loss_coef = 0.001
    router_jitter_noise = 0.0        # default off
    sliding_window = None             # no sliding window (Mistral-7B có, Mixtral không)
    rope_theta = 1000000.0
    vocab_size = 32000
```

Active params per token:

- Attention: ~0.5B (GQA 4:1 với 4096 hidden)
- 2 experts × FFN (~5.6B mỗi) = ~11.2B
- Embedding + lm_head + norm: ~1.2B
- Total: ~12.9B

Total params: ~46.7B.

## Cấu trúc class

```
modeling_mixtral.py (704 dòng)
├── MixtralExperts                      # 3D weight, @use_experts_implementation
├── MixtralTopKRouter                   # Linear + softmax + topk
├── MixtralSparseMoeBlock               # Wrap router + experts + jitter
├── MixtralRMSNorm
├── MixtralRotaryEmbedding
├── MixtralAttention                    # GQA
├── MixtralDecoderLayer                 # attn + sparse_moe_block
├── MixtralPreTrainedModel
├── MixtralModel
├── MixtralForCausalLM
├── MixtralForSequenceClassification
├── MixtralForQuestionAnswering
├── MixtralForTokenClassification
└── load_balancing_loss_func             # Helper
```

## `MixtralExperts`

```python
@use_experts_implementation
class MixtralExperts(nn.Module):
    """Collection of expert weights stored as 3D tensors."""

    def __init__(self, config: MixtralConfig):
        super().__init__()
        self.num_experts = config.num_local_experts
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

Phân tích:

**Decorator**: `@use_experts_implementation` (default flags: `is_concatenated=True, is_transposed=False, has_bias=False, has_gate=True`).

**Weight layout**:

- `gate_up_proj`: `(E=8, 2*14336, 4096)`. Concatenated layout.
- `down_proj`: `(E=8, 4096, 14336)`.

Total per expert: `(2*14336*4096) + (4096*14336) = 117M + 58.7M = ~175M`. Tám expert: ~1.4B. Cộng attention/norm/embed: ~5.6B per expert FFN, ~12.9B active total. (Chi tiết params calc phụ thuộc kiểu count.)

Đợi tính lại: 1 expert có `gate_up_proj` chừng 117M + `down_proj` chừng 58.7M = 175M params per expert. 8 expert = 1.4B in MoE only. Total model 46.7B chia hết bao gồm 32 layer attention + 32 layer MoE + embed + norm.

**Forward (eager path)**:

1. `expert_mask = one_hot(top_k_index, E)`: shape `(N, k, E)`. 1 ở vị trí token chọn expert.
2. `permute(2, 1, 0)`: shape `(E, k, N)`. Thuận tiện iterate expert.
3. `expert_hit`: list các expert nào được chọn (ít nhất 1 token).
4. Loop expert. Với mỗi expert:
   - `top_k_pos, token_idx = where(expert_mask[expert_idx])`: position của token đã chọn expert này.
   - `current_state = hidden_states[token_idx]`: gather tokens.
   - `gate, up = linear(...).chunk(2)`: SwiGLU forward.
   - `current_hidden = act(gate) * up`.
   - `linear(...)`: down projection.
   - Multiply routing weight `top_k_weights[token_idx, top_k_pos]`.
   - `index_add_`: scatter back.

Đây là **eager** implementation. Khi `config._experts_implementation = "grouped_mm"`, decorator dispatch về `grouped_mm_experts_forward` (xem Phần 2).

## `MixtralTopKRouter`

```python
class MixtralTopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_local_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits.float(), dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices
```

(`class MixtralTopKRouter`.)

Đã giải thích kỹ ở Phần 1 Chương 2. Đặc trưng:

- Linear `(E, hidden)` không bias.
- Softmax ở fp32.
- Top-k = 2.
- Renormalize weight sum = 1.
- Trả 3 thứ: logits (cho aux loss), scores (cho combine), indices (cho dispatch).

## `MixtralSparseMoeBlock`

```python
class MixtralSparseMoeBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.jitter_noise = config.router_jitter_noise
        self.gate = MixtralTopKRouter(config)
        self.experts = MixtralExperts(config)

    def forward(self, hidden_states):
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        if self.training and self.jitter_noise > 0:
            hidden_states *= torch.empty_like(hidden_states).uniform_(
                1.0 - self.jitter_noise, 1.0 + self.jitter_noise
            )
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        _, top_k_weights, top_k_index = self.gate(hidden_states)
        hidden_states = self.experts(hidden_states, top_k_index, top_k_weights)
        hidden_states = hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return hidden_states
```

(`class MixtralSparseMoeBlock`.)

Quy trình:

1. Optional jitter noise (train only).
2. Flatten batch+seq.
3. Router → (top_k_weights, top_k_index).
4. Experts(hidden_states, top_k_index, top_k_weights) → output.
5. Reshape về `(batch, seq, hidden)`.

**Note**: SparseMoeBlock không trả `router_logits` ra ngoài trong return signature. Để aux loss work, router_logits phải được captured khác. Mixtral dùng `OutputRecorder` (HF abstraction) bắt output ở model level.

## `MixtralAttention`

Standard GQA với RoPE, không có gì đặc biệt MoE. Đọc Phần 1 chuỗi bài giảng nền nếu chưa quen.

```python
class MixtralAttention(nn.Module):
    def __init__(self, config, layer_idx):
        ...
        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * head_dim, config.hidden_size, bias=False)
        ...
```

GQA 4:1 (32 Q head, 8 KV head). Tiết kiệm KV cache ~4x so với MHA.

## `MixtralDecoderLayer`

```python
class MixtralDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.self_attn = MixtralAttention(config, layer_idx)
        self.block_sparse_moe = MixtralSparseMoeBlock(config)
        self.input_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = MixtralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(self, hidden_states, ...):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attn_output, ... = self.self_attn(hidden_states, ...)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.block_sparse_moe(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, ...
```

(`class MixtralDecoderLayer`.)

Identical với Llama decoder layer **trừ** `self.mlp` được thay bằng `self.block_sparse_moe`. Đây là pattern lặp lại cho mọi model trong Phần 3.

**Mọi 32 layer đều là MoE**. Không có alternate dense/sparse như Switch.

## `MixtralForCausalLM`

```python
class MixtralForCausalLM(MixtralPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan = {"lm_head": "colwise_allgather"}

    def __init__(self, config):
        super().__init__(config)
        self.model = MixtralModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.router_aux_loss_coef = config.router_aux_loss_coef
        self.num_experts = config.num_local_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.post_init()

    def forward(self, input_ids, ..., output_router_logits=None, ...):
        outputs = self.model(input_ids, ..., output_router_logits=output_router_logits, ...)
        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        aux_loss = None
        if labels is not None:
            loss = self.loss_function(logits, labels, vocab_size=self.config.vocab_size, ...)
            if output_router_logits:
                aux_loss = load_balancing_loss_func(
                    outputs.router_logits, self.num_experts, self.num_experts_per_tok, attention_mask,
                )
                loss += self.router_aux_loss_coef * aux_loss.to(loss.device)

        return MoeCausalLMOutputWithPast(
            loss=loss, aux_loss=aux_loss, logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            router_logits=outputs.router_logits,
        )
```

(`class MixtralForCausalLM.forward`.)

Khác Llama:

- Aux loss compute ở đây (nếu `output_router_logits=True`).
- Output type là `MoeCausalLMOutputWithPast` (có `aux_loss` + `router_logits`).
- Config có thêm `router_aux_loss_coef`, `num_experts`, etc.

Còn lại (lm_head tied, generate, beam search) cùng Llama.

## Khi đọc các chương Phần 3 sau

Mỗi model so với Mixtral khác:

- **Switch (chương 3)**: encoder-decoder T5, top-1, có capacity, SwitchTransformersExperts dùng `ModuleDict` (kế thừa pattern T5).
- **DeepSeek-V3 (chương 4)**: sigmoid router, group routing, shared expert, aux-free bias.
- **Qwen3-MoE (chương 5)**: gần Mixtral nhất, k=8 thay vì 2.
- **GPT-OSS (chương 6)**: `is_transposed=True`, `has_bias=True`, clamp gate.
- **OLMoE (chương 7)**: gần Mixtral, k=8.
- **JetMoE (chương 8)**: thêm MoA (Mixture of Attention).
- **Jamba (chương 9)**: Mamba block + MoE block alternate.
- **NLLB-MoE (chương 10)**: encoder-decoder, capacity, expert dropout.
- **PhiMoE (chương 11)**: Phi-style, k=2.

## Pitfall

**1. Mixtral 8x7B name confusion**: không phải 8 model 7B mỗi cái. Là 1 model với 8 expert trong FFN, tổng 46.7B. Tên "8x7B" lừa đặt.

**2. Mixtral không có sliding window**. Mistral-7B base có. Khi fine-tune Mixtral, không apply Mistral attention pattern.

**3. Aux loss của Mixtral coef 0.001**: nhỏ. Một số fork tăng lên 0.01 để load balance tốt hơn, nhưng làm task loss khó giảm.

**4. Quên `output_router_logits=True` ở train**: aux loss = 0, expert collapse nhanh. Phải force.

**5. EP với Mixtral**: 8 expert, nếu deploy 4 GPU thì mỗi GPU 2 expert. Số GPU phải chia hết `num_experts`.

Chương sau ta đọc Switch Transformers.
