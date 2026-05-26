---
title: Glossary
---

# Glossary

Thuật ngữ MoE thường gặp, kèm tham chiếu chương đào sâu.

## A

**Active params**: số parameter thực sự được dùng cho **một token** mỗi forward. So với total params. Phần 0 Chương 2.

**ALL_EXPERTS_FUNCTIONS**: singleton `ExpertsInterface` chứa các backend (batched_mm, grouped_mm). Phần 2 Chương 3.

**All-to-all**: collective communication primitive permute data giữa các rank. EP dùng. Phần 4 Chương 2.

**Aux loss (auxiliary loss)**: loss bổ sung phạt expert distribution không đều. Phần 1 Chương 4.

**Aux-free balancing**: bias adjustment kiểu DeepSeek-V3, không cần aux loss. Phần 1 Chương 4.

## B

**`batched_mm_experts_forward`**: backend MoE dispatch dùng `einsum`/expand, không sort. Phần 2 Chương 4.

**Bias correction**: bias `(num_experts,)` cộng vào router score cho dispatch quyết định, update qua callback. DeepSeek-V3. Phần 1 Chương 4.

**Block-sparse matmul**: kernel matmul cho dispatch token vào expert nhóm. Megablocks. Phần 4 Chương 6.

## C

**Capacity factor**: tỉ lệ token tối đa mỗi expert nhận. `1.0` = expected, `1.25` = 25% margin. Switch. Phần 1 Chương 5.

**`config._experts_implementation`**: string báo backend nào (`"eager"`, `"batched_mm"`, `"grouped_mm"`). Phần 2 Chương 3.

**Continuous batching**: serving strategy pack request đang ở phase khác (prefill, decode) vào cùng batch. Phần 4 Chương 5.

**Coarse-grained MoE**: số expert ít (≤ 16), mỗi expert lớn. Mixtral. Phần 1 Chương 6.

## D

**DeepSeek-V3**: model MoE state-of-the-art 2024, 671B/37B, sigmoid + bias adjust + shared. Phần 3 Chương 4.

**Decoder layer (MoE)**: layer Transformer với attention + sparse_moe_block thay vì FFN. Phần 3 Chương 2.

**Dispatch**: bước route token đến đúng expert. Phần 1 Chương 3.

**Dropless MoE**: paradigm không drop token, dùng `grouped_mm`. Mixtral, DeepSeek, Qwen3. Phần 1 Chương 5.

## E

**EP (Expert parallelism)**: phân phối expert qua nhiều GPU. Phần 4 Chương 2.

**Expert**: sub-module (MLP) trong MoE. Phần 0 Chương 3.

**Expert capacity**: số token tối đa expert nhận. Phần 1 Chương 5.

**Expert-choice routing**: expert chọn token (đảo ngược token-choice). V-MoE. Phần 1 Chương 3.

**Expert dropout**: drop entire expert ngẫu nhiên ở train. NLLB-MoE. Phần 3 Chương 10.

**`ExpertsInterface`**: HF registry cho backend dispatch (parallel với `AttentionInterface`). Phần 2 Chương 3.

## F

**Fine-grained MoE**: số expert nhiều (≥ 64), mỗi expert nhỏ. DeepSeek, Qwen3, OLMoE. Phần 1 Chương 6.

**FP8 (E4M3, E5M2)**: 8-bit float quant cho weight + activation. DeepSeek-V3 train. Phần 4 Chương 4.

**FSDP**: Fully Sharded Data Parallel, shard parameter theo data-parallel rank. Phần 4 Chương 3.

## G

**Gate (router)**: linear layer xuất router_logits từ hidden_states. Phần 1 Chương 2.

**Gate logits**: output raw của router, trước softmax/sigmoid. Phần 1 Chương 2.

**GPT-OSS**: model OpenAI 2025 với MXFP4 native + clamp gate. Phần 3 Chương 6.

**Group routing**: chia expert thành nhóm, route hai tầng. DeepSeek-V3. Phần 1 Chương 3.

**`grouped_mm_experts_forward`**: backend MoE dispatch dùng `torch._grouped_mm`. Phần 2 Chương 4.

## I

**`is_concatenated`**: flag decorator báo gate_up_proj concat (`True`) hay tách (`False`). Phần 2 Chương 3.

**`is_transposed`**: flag báo weight `(E, in, out)` (`True`) hay `(E, out, in)` (`False`). Phần 2 Chương 3.

## J

**Jamba**: hybrid Mamba + Transformer + MoE (AI21). Phần 3 Chương 9.

**JetMoE**: MoA (Mixture of Attention) + MoE FFN. Phần 3 Chương 8.

**Jitter noise**: nhân `hidden_states` với uniform noise trước router ở train. Switch, Mixtral. Phần 1 Chương 2.

## K

**KV cache (cho MoE)**: cache K/V cho attention. Không đổi từ dense, nhưng tương tác với MLA (DeepSeek). Phần 4 Chương 5.

## L

**Load balancing**: chiến lược ngăn expert collapse. Phần 1 Chương 4.

**`load_balancing_loss_func`**: helper compute aux loss từ tuple router_logits. Phần 2 Chương 5.

## M

**Megablocks**: kernel CUDA cho dropless MoE (Stanford/MosaicML 2023). Phần 4 Chương 6.

**Mixtral**: model MoE baseline (Mistral AI 2023), 8 expert top-2. Phần 3 Chương 2.

**MLA (Multi-head Latent Attention)**: attention với compressed KV (DeepSeek). Phần 3 Chương 4.

**MoA (Mixture of Attention)**: routing attention heads. JetMoE. Phần 3 Chương 8.

**ModuleDict experts**: legacy layout, ModuleDict với 1 module per expert. Switch, Jamba. Phần 1 Chương 6.

**MXFP4**: 4-bit microscaling FP. GPT-OSS native. Phần 4 Chương 4.

## N

**`n_group`, `topk_group`**: config DeepSeek-V3 group routing. Phần 1 Chương 3.

**`norm_topk_prob`**: flag config nếu renormalize topk weights sum=1. Phần 1 Chương 2.

**`num_experts_per_tok`**: k của top-k routing. Phần 0 Chương 3.

**`num_local_experts`**: số expert routed (excluding shared). Phần 0 Chương 3.

## O

**OLMoE**: open-data MoE (AI2). Phần 3 Chương 7.

**`output_router_logits`**: flag forward trả router_logits ra cho aux loss. Phần 2 Chương 5.

## P

**`PreTrainedModel` flags**: `_supports_*`, `_tp_plan`, `_ep_plan`. Phần 4 Chương 3.

**PhiMoE**: small-scale MoE từ Microsoft. Phần 3 Chương 11.

## R

**Renormalize topk**: chia topk weights cho sum để sum=1 ở top-k. Phần 1 Chương 2.

**Router**: linear + activation chọn expert. Phần 1 Chương 2.

**RouterParallel**: HF abstraction cho EP-aware router. Phần 4 Chương 2.

**`router_aux_loss_coef`**: hyper-parameter scale aux loss. Phần 1 Chương 4.

**`router_jitter_noise`**: hyper-parameter cho jitter. Phần 1 Chương 2.

**`router_z_loss_coef`**: hyper-parameter cho z-loss. Phần 1 Chương 4.

**`routed_scaling_factor`**: scale routing weights (DeepSeek-V3 = 2.5). Phần 1 Chương 4.

## S

**Sentinel**: expert id ngoài range của local GPU (cho EP). Phần 4 Chương 2.

**Shared expert**: expert luôn được dùng cho mọi token. DeepSeek. Phần 1 Chương 6.

**Sigmoid routing**: dùng sigmoid thay softmax. DeepSeek-V3. Phần 1 Chương 2.

**Sparsity ratio**: `active / total params`. Phần 0 Chương 2.

**SparseMoeBlock**: module wrap router + experts. Phần 3 Chương 2.

**Switch Transformer**: top-1 + capacity, encoder-decoder T5. Phần 3 Chương 3.

## T

**Token-choice routing**: token chọn expert (default). Phần 1 Chương 3.

**Token dropping**: drop token vượt capacity. Switch. Phần 1 Chương 5.

**Top-1, top-k**: chọn 1 hoặc k expert mỗi token. Phần 1 Chương 3.

**Total params**: tổng parameter trong model, kể cả expert không active. Phần 0 Chương 2.

**TP (Tensor parallel)**: shard tensor dimension qua GPU. Phần 4 Chương 3.

## U

**`use_experts_implementation`**: decorator rewrite expert class để dispatch qua interface. Phần 2 Chương 3.

## Z

**Z-loss (router z-loss)**: phạt large logit magnitude. Switch, OLMoE. Phần 1 Chương 4.

**3D weight tensor**: layout modern cho expert weight `(E, out, in)`. Mixtral và post-2024. Phần 1 Chương 6.
