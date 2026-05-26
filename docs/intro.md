---
title: Giới thiệu chuỗi bài giảng
slug: /intro
---

# Giới thiệu

Mixture of Experts (MoE) đã trở thành kiến trúc chủ đạo của các LLM frontier 2023-2025: Mixtral, Gemini, GPT-4 (rumored), Claude (rumored), DeepSeek-V3, Qwen-MoE, GPT-OSS. Lý do đơn giản: với cùng compute, MoE có nhiều parameter hơn dense; với cùng parameter, MoE rẻ hơn dense khi inference. Hai trade-off này đủ để công nghiệp đầu tư hàng tỉ USD vào MoE.

Nhưng MoE không phải một thiết kế đơn nhất. Mỗi paper, mỗi model có lựa chọn riêng về router, expert dispatch, load balancing. Mỗi quyết định mở ra một paradigm. Chuỗi bài giảng này đọc 10 model MoE trong HuggingFace `transformers`, trải nghiệm thiết kế qua source code thật.

## Đối tượng

Bạn nên đã biết:

- PyTorch cơ bản, viết được `nn.Module` với forward đơn giản.
- Kiến trúc Transformer dense (attention, MLP, layer norm, residual).
- Đã đọc một file `modeling_*.py` của HuggingFace ít nhất một lần (Llama là phổ biến nhất).

Bạn **không** cần biết MoE từ trước. Phần 1 sẽ xây toàn bộ intuition.

Bạn sẽ học được:

- Cấu trúc một router: gate logits, top-k, jitter noise, normalization (softmax vs sigmoid).
- Routing strategy: token-choice vs expert-choice, top-1 vs top-k, group routing.
- Load balancing: auxiliary loss, z-loss, bias adjustment kiểu DeepSeek (aux-free).
- Expert capacity, token dropping, dropless variants.
- Shared experts (DeepSeek), fine-grained experts (DeepSeek/OLMoE), expert dropout (NLLB-MoE).
- HuggingFace MoE infrastructure: `integrations/moe.py`, `ExpertsInterface`, `batched_mm` vs `grouped_mm`.
- Walkthrough 10 model: Mixtral, Switch Transformers, DeepSeek-V3, Qwen3-MoE, GPT-OSS, OLMoE, JetMoE, Jamba, NLLB-MoE, PhiMoE.
- Cross-cutting: expert parallelism (EP), tensor parallel (TP) tương tác với EP, MXFP4 và FP8 quantization, continuous batching với MoE.

## Triết lý đọc source

Cùng triết lý "một model một file" như phần còn lại của transformers. Mỗi model MoE có một `modeling_*moe*.py` chứa class `*Router`, `*Experts`, `*SparseMoeBlock`, `*MoE` cùng decoder layer. Có hai điểm đặc biệt:

1. **Hạ tầng chia sẻ** đã được tách thành `src/transformers/integrations/moe.py`. Hầu hết model 2024-2025 dùng decorator `@use_experts_implementation` để dispatch expert computation qua `batched_mm` hoặc `grouped_mm`. Hiểu file này giúp đọc nhanh mọi model mới.

2. **Modular file**: nhiều model MoE (Qwen3-MoE, OLMoE, ...) có `modular_*.py` ngắn kế thừa từ một model gốc (Mixtral, Llama). File `modeling_*.py` generated từ đó. Đọc modular trước thấy được "diff" so với base; đọc modeling thấy code phẳng đầy đủ.

## Phương pháp

Mỗi chương đều tuân theo cấu trúc:

1. **Trực giác trước**: vấn đề là gì, vì sao cần giải, ý tưởng cốt lõi.
2. **Toán hoặc đặc tả**: derive hoặc trình bày chính xác cái gì đang được implement.
3. **Walkthrough source**: mở file thật, đọc từng đoạn, gắn trở lại với toán.
4. **Pitfall và edge case**: chỗ dễ sai khi tự viết model hoặc khi debug.

Mọi đoạn code trích dẫn đều từ thư viện thật. Nếu bạn clone `huggingface/transformers` cùng version, sẽ tìm thấy các dòng đó nguyên văn.

## Cấu trúc các phần

Phần 0 giới thiệu Mixture of Experts: sparse vs dense, scaling laws, lịch sử vắn tắt từ Adaptive Mixtures of Local Experts (1991) đến Mixtral (2023). Định nghĩa thuật ngữ và roadmap.

Phần 1 xây foundations: router anatomy, routing strategies (token-choice, expert-choice), load balancing (auxiliary loss, z-loss, bias adjustment), expert capacity và token dropping, shared experts và fine-grained partitioning.

Phần 2 walkthrough `src/transformers/integrations/moe.py`: `ExpertsInterface`, `ALL_EXPERTS_FUNCTIONS`, decorator `use_experts_implementation`, `batched_mm_experts_forward` vs `grouped_mm_experts_forward`, helper `load_balancing_loss_func`.

Phần 3 đọc 10 model MoE: Mixtral (baseline), Switch Transformers (encoder-decoder + top-1), DeepSeek-V3 (state-of-the-art), Qwen3-MoE (modern infra), GPT-OSS (production + MXFP4), OLMoE (open recipe), JetMoE (MoA + MoE), Jamba (Mamba + MoE), NLLB-MoE (translation), PhiMoE (small-scale).

Phần 4 cross-cutting: expert parallelism (EP) với `RouterParallel` và sentinels, tensor parallel cho MoE, MXFP4 quantization (`integrations/mxfp4.py`), FP8 (`integrations/finegrained_fp8.py`), continuous batching và serving, training recipe.

Phần 5 so sánh thiết kế: matrix router, balancing, expert design ngang 10 model. Decision tree khi nào chọn MoE vs dense, top-1 vs top-k, shared vs fine-grained.

## Liên kết với chuỗi bài giảng nền

Repo này tập trung vào MoE. Các chủ đề attention, KV cache, generate, conventions của `PreTrainedModel` được giả định là người đọc đã quen ở mức cơ bản. Khi nhắc đến những chủ đề đó (ví dụ MoE tương tác với KV cache, hoặc `from_pretrained` apply TP plan), chỉ trình bày phần liên quan đến MoE, không giải thích lại nền tảng.

## Source reference

Mọi trích dẫn dựa trên branch `main` của `huggingface/transformers` tại thời điểm biên soạn. API MoE đang tiến hoá nhanh (`use_experts_implementation` decorator mới có cuối 2024). Khi đọc, đối chiếu với version bạn đang dùng nếu khác.

Sẵn sàng, vào Phần 0.
