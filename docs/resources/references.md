---
title: References
---

# References

Tài liệu để đào sâu hơn ngoài source code transformers.

## Source code chính

- HuggingFace transformers: https://github.com/huggingface/transformers
- File trọng tâm cho MoE:
  - `src/transformers/integrations/moe.py` (ExpertsInterface, batched_mm, grouped_mm)
  - `src/transformers/integrations/mxfp4.py` (MXFP4 quantization)
  - `src/transformers/integrations/finegrained_fp8.py` (FP8 quantization)
  - `src/transformers/models/*/modeling_*moe*.py` (10 model walkthrough)

## Papers nền tảng

### Foundational

- **Adaptive Mixtures of Local Experts** (Jacobs et al., 1991). MoE original.
- **Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer** (Shazeer et al., 2017). Modern MoE foundation.
- **GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding** (Lepikhin et al., 2020). Google T5-MoE.

### Switch / capacity

- **Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity** (Fedus et al., 2021).
- **ST-MoE: Designing Stable and Transferable Sparse Expert Models** (Zoph et al., 2022). Z-loss.

### Expert-choice

- **Mixture-of-Experts with Expert Choice Routing** (Zhou et al., 2022). EC routing.
- **Scaling Vision with Sparse Mixture of Experts** (Riquelme et al., 2021). V-MoE.

### Modern LLM MoE

- **Mixtral of Experts** (Jiang et al., 2024). Mixtral 8x7B.
- **DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model** (DeepSeek-AI, 2024).
- **DeepSeek-V3 Technical Report** (DeepSeek-AI, 2024).
- **Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts** (Wang et al., 2024). DeepSeek bias adjustment.
- **OLMoE: Open Mixture-of-Experts Language Models** (Muennighoff et al., 2024).
- **JetMoE: Reaching Llama2 Performance with 0.1M Dollars** (Shen et al., 2024).
- **Jamba: A Hybrid Transformer-Mamba Language Model** (AI21 Labs, 2024).
- **No Language Left Behind: Scaling Human-Centered Machine Translation** (NLLB Team, 2022).

### Optimization / serving

- **MegaBlocks: Efficient Sparse Training with Mixture-of-Experts** (Gale et al., 2023).
- **Efficient Memory Management for Large Language Model Serving with PagedAttention** (Kwon et al., 2023). vLLM paper.
- **MoE-LightFlow: Hardware-Efficient Inference of Sparse MoE Models** (assumption).

### Quantization

- **MX (Microscaling) Compute** (Open Compute Project, 2023). MXFP4 spec.
- **FP8 Formats for Deep Learning** (NVIDIA, 2022). E4M3, E5M2.
- **FP8-LM: Training FP8 Large Language Models** (Peng et al., 2023).

### Mamba / SSM

- **Mamba: Linear-Time Sequence Modeling with Selective State Spaces** (Gu, Dao, 2023).
- **Mamba-2** (Dao, Gu, 2024).

## Blog posts và lectures

- HuggingFace blog series về MoE: https://huggingface.co/blog
  - "Mixture of Experts Explained"
  - "Welcoming Mixtral"
  - "Inference with Mixtral"
- Sasha Rush, "MoE Tutorial": YouTube.
- Stanford CS25 "Transformers United" lectures về MoE.

## PyTorch tài liệu

- `torch.nn.functional.scaled_dot_product_attention`: standard attention.
- `torch._grouped_mm` / `torch.nn.functional.grouped_mm`: MoE expert dispatch.
- DTensor API: distributed tensor abstraction cho TP/EP.
- FSDP: https://pytorch.org/docs/stable/fsdp.html
- `torch.compile`: tutorial.

## Other libraries trong ecosystem

- **vLLM**: https://github.com/vllm-project/vllm. Production serving với MoE.
- **TGI** (Text Generation Inference): https://github.com/huggingface/text-generation-inference.
- **DeepSpeed**: https://github.com/microsoft/DeepSpeed. FSDP + EP + ZeRO.
- **Megatron-LM**: https://github.com/NVIDIA/Megatron-LM. TP + EP + PP cho train scale lớn.
- **PEFT**: https://github.com/huggingface/peft. LoRA cho MoE.
- **OLMo**: https://github.com/allenai/OLMo. Training code cho OLMoE (public).
- **Megablocks**: https://github.com/databricks/megablocks. Block-sparse kernel.

## Practical guides

- **HuggingFace MoE docs**: https://huggingface.co/docs/transformers/moe
- **DeepSeek-V3 README**: https://huggingface.co/deepseek-ai/DeepSeek-V3
- **Mixtral README**: https://huggingface.co/mistralai/Mixtral-8x7B-v0.1
- **vLLM MoE guide**: vLLM docs.

## Đọc tiếp khi cần

Sau chuỗi này:

1. **Implement MoE từ đầu**: viết một SparseMoeBlock 50 dòng, train trên small dataset (10M params). Hiểu sâu hơn lý thuyết.
2. **Fine-tune một MoE production**: pick Mixtral 8x7B hoặc Qwen3-30B, fine-tune với LoRA cho task riêng.
3. **Implement custom routing**: viết expert-choice routing cho vision, hoặc group routing tự chế.
4. **Profile MoE inference**: dùng vLLM hoặc TGI, measure latency, expert utilization, optimize.
5. **Train MoE từ scratch**: nếu có cluster. OLMoE recipe public, có thể follow.

## Đề xuất lộ trình tiếp theo

1. Đọc chuỗi bài giảng nền (attention, KV cache, generate, conventions) nếu chưa.
2. Fine-tune Mixtral 8x7B với LoRA trên dataset của bạn.
3. Profile vLLM serving Mixtral, measure throughput.
4. Đọc paper DeepSeek-V3 chi tiết.
5. Đọc papers Megablocks và FP8-LM.
6. Tham gia vào codebase HF transformers (issue, PR cho model MoE mới).

Mỗi task trên là một dự án 2-4 tuần. Hoàn thành 2-3 cái là ở level advanced ML engineer cho MoE.

## Lời cảm ơn cuối

Chuỗi này dựa trên codebase transformers ở thời điểm viết. API MoE đang tiến hoá nhanh (`use_experts_implementation` mới có cuối 2024). Citation có thể stale sau vài tháng. Tinh thần cốt lõi (router, dispatch, balancing, EP, quantization) sẽ stable lâu hơn.

Chúc bạn đọc source code MoE với tự tin hơn.
