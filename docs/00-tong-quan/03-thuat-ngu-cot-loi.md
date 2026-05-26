---
title: Thuật ngữ cốt lõi
---

# Thuật ngữ cốt lõi

Chương này định nghĩa nhanh mọi thuật ngữ MoE sẽ gặp lặp lại. Mục đích: khi đọc Phần 1 trở đi, không phải dừng lại tra từ. Mỗi định nghĩa kèm chỉ chương sẽ đi sâu.

## Cấu trúc cơ bản

**Mixture of Experts (MoE)**: kiến trúc thay thế một module dense (thường là Feed Forward Network trong Transformer) bằng một bộ nhiều **expert** cộng với một **router** quyết định token nào đi qua expert nào.

**Expert**: một sub-module có cùng kiến trúc (thường là MLP với SwiGLU), nhưng parameter độc lập. Một MoE layer có 8 đến 256+ expert. Phần 1 Chương 6.

**Router** (còn gọi là **gate**): module nhỏ (thường là một linear layer) nhận `hidden_states` và xuất ra `router_logits` shape `(batch, seq, num_experts)`. Phần 1 Chương 2.

**Gate logits**: output của router trước khi softmax. Phần 1 Chương 2.

**Routing weights**: gate logits sau khi softmax (hoặc sigmoid), normalize thành probability. Phần 1 Chương 2.

**Sparse MoE block**: module bao gồm router + experts + forward logic dispatch token. Trong code HF thường tên `*SparseMoeBlock` (Mixtral, Qwen) hoặc `*MoE` (DeepSeek).

## Routing decisions

**Top-k routing**: chọn k expert có routing weight cao nhất cho mỗi token. `k=1` là top-1 (Switch), `k=2` là top-2 (Mixtral, Jamba), `k=8` (DeepSeek-V3, OLMoE). Phần 1 Chương 3.

**Token-choice routing**: token chọn expert. Default cho hầu hết model. Phần 1 Chương 3.

**Expert-choice routing**: expert chọn token (đảo ngược). Mỗi expert chọn top-N token có routing weight cao nhất với expert đó. Đảm bảo load balance tuyệt đối. Dùng trong V-MoE (vision), không phổ biến trong LLM. Phần 1 Chương 3.

**Group routing (n_group, topk_group)**: chia expert thành nhóm, mỗi token chỉ chọn expert trong subset nhóm. Giảm communication overhead. Dùng trong DeepSeek-V3. Phần 1 Chương 3.

**Jitter noise**: thêm random multiplicative noise vào `hidden_states` trước khi tính router_logits, chỉ ở train mode. Mục đích: encourage exploration, tránh router thoái hoá. Switch Transformer. Phần 1 Chương 2.

## Load balancing

**Load balance**: phân bổ token đều giữa expert. Mong muốn: mỗi expert nhận xấp xỉ `(batch * seq * k) / num_experts` token. Phần 1 Chương 4.

**Auxiliary loss (aux loss)**: loss bổ sung khuyến khích router phân bổ đều. Công thức Mixtral:

```
aux_loss = num_experts * sum(f_i * P_i)
```

trong đó `f_i` là fraction token đi đến expert `i`, `P_i` là average routing weight cho expert `i`. Minimize loss này đẩy `f_i` và `P_i` về uniform. Phần 1 Chương 4.

**Z-loss (router z-loss)**: loss thứ hai phạt large logit để stabilize numerical. Switch Transformer, ST-MoE. Phần 1 Chương 4.

**Aux-free balancing**: phương pháp DeepSeek-V3 không dùng auxiliary loss. Mỗi expert có một **bias** dynamic, tự động tăng nếu expert đó underutilized và giảm nếu overutilized. Phần 1 Chương 4.

## Capacity và dropping

**Expert capacity**: số token tối đa một expert được phép nhận trong một batch. Capacity factor = 1.0 nghĩa là mỗi expert nhận đúng `(batch * seq * k) / num_experts` token. Switch dùng capacity factor 1.0 hoặc 1.25. Phần 1 Chương 5.

**Token dropping**: nếu router gửi quá nhiều token đến một expert vượt capacity, các token "thừa" bị drop (chỉ qua residual, không qua expert). Switch Transformer. Phần 1 Chương 5.

**Dropless MoE**: không dùng capacity, mọi token đều được expert xử lý. Cần kernel custom (megablocks) hoặc `grouped_mm` để hiệu quả. Mixtral, DeepSeek, hầu hết model 2024+. Phần 1 Chương 5.

## Cấu trúc expert đặc biệt

**Shared expert**: một expert luôn được dùng cho mọi token, không qua router. Tách kiến thức "general" khỏi kiến thức "specialized". DeepSeek-V2, V-V3. Phần 1 Chương 6.

**Fine-grained experts**: thay vì 8 expert lớn, dùng 256 expert nhỏ. Mỗi expert chuyên hơn, total params giữ nguyên nhưng routing mịn hơn. DeepSeek-V3, OLMoE. Phần 1 Chương 6.

**Expert dropout**: ở training, drop entire expert với xác suất p. Regularization mạnh, ngăn router bị "stuck" với một subset expert. NLLB-MoE. Phần 3 Chương 10.

## Infrastructure

**Expert parallelism (EP)**: phân phối expert qua nhiều GPU. Mỗi GPU giữ một subset expert. Router output quyết định token nào gửi đến GPU nào (all-to-all communication). Phần 4 Chương 2.

**RouterParallel**: HF abstraction quản lý EP, xử lý sentinel cho expert id ngoài range của GPU hiện tại. Phần 4 Chương 2.

**ExpertsInterface**: HF abstraction (parallel với `AttentionInterface`) cho phép swap expert implementation: native PyTorch, `batched_mm`, `grouped_mm`. Phần 2 Chương 3.

**batched_mm**: implementation dùng einsum/expand cho batch nhỏ. Phần 2 Chương 4.

**grouped_mm**: implementation dùng `torch._grouped_mm` (PyTorch 2.9+) cho batch lớn, hiệu quả hơn `batched_mm`. Yêu cầu SM80+ GPU. Phần 2 Chương 4.

**MXFP4**: 4-bit microscaling float format, dùng cho weight expert trong GPT-OSS. Phần 4 Chương 4.

**FP8 (E4M3, E5M2)**: 8-bit float, dùng cho training và inference MoE. Phần 4 Chương 4.

## Output dataclass

**MoE output**: ngoài `logits`, `hidden_states`, `past_key_values`, MoE model trả thêm `router_logits` (tuple per layer) để compute auxiliary loss ở train. Phần 3 Chương 2.

**`output_router_logits`**: flag forward bật để model trả `router_logits`. Default `False` ở inference, `True` ở train.

## Decoder layer

**Sparse layer**: layer có MoE thay vì dense FFN.

**Dense layer**: layer giữ FFN dense. Một số model alternate sparse/dense (Switch dùng pattern này; cũng có model giữ vài layer đầu dense).

**Layer pattern**: thứ tự sparse/dense theo từng layer. Trong Jamba, pattern còn phức tạp hơn vì có Mamba block. Phần 3 Chương 9.

## Liên kết với Transformer

**FFN ratio**: tỉ lệ `intermediate_size / hidden_size`. Dense thường 4x (Llama là 8/3 ≈ 2.67x với SwiGLU). MoE fine-grained dùng nhỏ hơn (DeepSeek 8x với 256 expert, mỗi expert chỉ 1/8 size of dense FFN).

**SwiGLU, GeGLU, ReGLU**: variants activation cho FFN/expert. Hầu hết MoE 2024+ dùng SwiGLU.

**Auxiliary outputs**: ngoài hidden_states, layer MoE có thể trả router_logits (cho aux loss) và expert_indices (cho profiling). Phần 3 Chương 2.

Chương sau là roadmap cho toàn series.
