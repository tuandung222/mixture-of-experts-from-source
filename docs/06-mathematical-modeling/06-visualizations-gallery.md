---
title: Visualizations gallery
---

# Visualizations gallery

Tất cả diagram + chart trong một chỗ. Dùng để slide, blog, hoặc reference quick. Mỗi diagram có caption giải thích.

## 1. MoE block tổng thể

```mermaid
flowchart TD
    Input["Hidden states<br/>shape: B, T, d"]
    Input --> Norm["RMSNorm"]
    Norm --> Router{"Router<br/>Linear E x d"}
    Router --> Logits["router_logits<br/>shape: N, E"]
    Logits --> TopK["Top-k Selection<br/>k=2 or 8"]
    TopK --> Indices["top_k_index<br/>shape: N, k"]
    TopK --> Weights["top_k_weights<br/>shape: N, k"]
    Indices --> Dispatch["Dispatch tokens to experts"]
    Weights --> Combine["Weighted combine"]
    Dispatch --> Experts["Experts<br/>3D weight: E, 2d_ff, d"]
    Experts --> Output1["Expert outputs<br/>shape: N x k, d"]
    Output1 --> Combine
    Combine --> ResidualAdd["plus input residual"]
    ResidualAdd --> Out["Output<br/>shape: B, T, d"]

    style Router fill:#fcc
    style TopK fill:#ffc
    style Experts fill:#cfc
    style Combine fill:#ccf
```

**Caption**: Forward pass một SparseMoeBlock. Router (đỏ) → top-k (vàng) → experts (xanh lá) → combine (xanh dương).

## 2. Token-choice vs Expert-choice routing

```mermaid
graph TB
    subgraph TC["Token-choice (default)"]
        T1["Token 1"] -->|"score 0.8"| TE1["Expert A"]
        T1 -->|"score 0.6"| TE2["Expert B"]
        T2["Token 2"] -->|"score 0.9"| TE3["Expert C"]
        T2 -->|"score 0.4"| TE1
        T3["Token 3"] -->|"score 0.7"| TE1
        T3 -->|"score 0.5"| TE3
        style TE1 fill:#fcc
        style TE2 fill:#cfc
        style TE3 fill:#cff
    end
    subgraph EC["Expert-choice (V-MoE)"]
        EA["Expert A"] -->|"top-2 tokens"| EAT1["T1, T3"]
        EB["Expert B"] -->|"top-2 tokens"| EBT["T1, T2"]
        EC1["Expert C"] -->|"top-2 tokens"| ECT["T2, T3"]
        style EA fill:#fcc
        style EB fill:#cfc
        style EC1 fill:#cff
    end
```

**Caption**: Token-choice: token chọn expert (default LLM). Expert-choice: expert chọn token (vision MoE).

## 3. Sparsity comparison

```mermaid
graph LR
    subgraph Dense["Llama-3-70B Dense"]
        D1["Total: 70B"]
        D2["Active: 70B<br/>100%"]
        style D2 fill:#fcc
    end
    subgraph Coarse["Mixtral 8x7B"]
        M1["Total: 47B"]
        M2["Active: 12.9B<br/>28%"]
        style M2 fill:#fcb
    end
    subgraph Fine["DeepSeek-V3"]
        DS1["Total: 671B"]
        DS2["Active: 37B<br/>5.5%"]
        style DS2 fill:#fab
    end
    subgraph Ultra["GPT-OSS-120B"]
        G1["Total: 117B"]
        G2["Active: 5.1B<br/>4.4%"]
        style G2 fill:#fac
    end
```

**Caption**: Sparsity progression 2023-2025. Density giảm dần, từ 100% (dense) xuống 4.4% (GPT-OSS-120B).

## 4. Group routing (DeepSeek-V3)

```mermaid
flowchart TB
    Token["Token vector<br/>d=7168"] --> Router["Router<br/>E x d weight"]
    Router --> Scores["Score: 256 experts<br/>after sigmoid"]
    Scores --> Bias["plus bias correction"]
    Bias --> Reshape["Reshape: 8 groups x 32 experts"]
    Reshape --> GroupScore["Compute group score<br/>= sum top-2 per group"]
    GroupScore --> GroupTopK["Top-4 groups<br/>by group score"]
    GroupTopK --> Mask["Mask out experts in non-selected groups"]
    Mask --> FinalTopK["Top-8 experts overall"]
    FinalTopK --> Indices["8 expert indices"]
    FinalTopK --> Weights["8 routing weights<br/>from original sigmoid"]
    Weights --> Renorm["Renormalize + scale 2.5x"]

    style Router fill:#fcc
    style Bias fill:#ffc
    style GroupScore fill:#cfc
    style FinalTopK fill:#ccf
    style Renorm fill:#fcf
```

**Caption**: DeepSeek-V3 two-stage routing. Score 256 expert → group → select 4 groups → top-8 trong selected groups. Bias cộng vào choice nhưng không vào output weights.

## 5. Expert parallelism dispatch

```mermaid
flowchart LR
    subgraph Before["Before all-to-all"]
        R0["Rank 0<br/>tokens 0-99"] --> P0["Routing decisions"]
        R1["Rank 1<br/>tokens 100-199"] --> P1["Routing decisions"]
        R2["Rank 2<br/>tokens 200-299"] --> P2["Routing decisions"]
        R3["Rank 3<br/>tokens 300-399"] --> P3["Routing decisions"]
    end
    P0 --> A2A["All-to-all dispatch"]
    P1 --> A2A
    P2 --> A2A
    P3 --> A2A
    A2A --> After
    subgraph After["After all-to-all"]
        After0["Rank 0<br/>tokens routed to experts 0-31"]
        After1["Rank 1<br/>tokens routed to experts 32-63"]
        After2["Rank 2<br/>tokens routed to experts 64-95"]
        After3["Rank 3<br/>tokens routed to experts 96-127"]
    end
    After0 --> Exp0["Run local experts"]
    After1 --> Exp1["Run local experts"]
    After2 --> Exp2["Run local experts"]
    After3 --> Exp3["Run local experts"]
    Exp0 --> A2A2["All-to-all gather"]
    Exp1 --> A2A2
    Exp2 --> A2A2
    Exp3 --> A2A2

    style A2A fill:#fcc
    style A2A2 fill:#fcc
```

**Caption**: EP forward pass. Hai all-to-all communication: dispatch (route token to local experts) và gather (collect output back to origin).

## 6. Load balancing aux loss flow

```mermaid
flowchart TD
    Router["Router<br/>router_logits N x E"] --> Softmax["Softmax to p<br/>routing probs"]
    Softmax --> TopK["Top-k indices"]
    Softmax --> P["Mean P_i = average prob"]
    TopK --> OneHot["one_hot to expert mask"]
    OneHot --> F["Fraction f_i = mean mask"]

    P --> Dot["Compute f dot P sum"]
    F --> Dot
    Dot --> AuxLoss["aux = E * sum(f * P)<br/>scalar"]
    AuxLoss --> Scale["times router_aux_loss_coef (0.001)"]
    Scale --> Total["plus to total loss"]
    CE["Cross-entropy loss"] --> Total

    style Softmax fill:#ffc
    style P fill:#cfc
    style F fill:#cfc
    style Dot fill:#fcc
    style AuxLoss fill:#fcf
    style Total fill:#ccf
```

**Caption**: Auxiliary loss computation flow. Sử dụng cả $\mathbf{f}$ (token fractions, không-diff) và $\mathbf{P}$ (mean probs, diff). Dot product penalize imbalance.

## 7. Bias adjustment as feedback control

```mermaid
flowchart LR
    Target["Target load<br/>f* = k/E"] --> Diff{f* - f_i}
    Actual[Current load f_i] --> Diff
    Diff -->|"sign(error)"| Update[Δb_i = lr × sign]
    Update --> NewBias[b_i ← b_i + Δb_i]
    NewBias --> AddToLogits[Add to router logits<br/>for choice only]
    AddToLogits --> Routing[Routing decision]
    Routing --> Forward[Forward pass]
    Forward --> CountLoad[Count tokens per expert]
    CountLoad --> Actual

    style Target fill:#cfc
    style Diff fill:#fcc
    style NewBias fill:#ccf
    style Forward fill:#ffc
```

**Caption**: DeepSeek-V3 bias adjustment as P-controller feedback loop. Error = target - actual. Sign-based update. No gradient.

## 8. Active vs Total params chart

```
Sparsity Progression Bar Chart (Active / Total ratio):

Llama-3-70B Dense:
████████████████████████████████████████ 100% (Active = Total)

Mixtral 8x7B:
███████████ 28% (12.9B / 46.7B)

Qwen3-30B-A3B:
████ 10% (3B / 30B)

DeepSeek-V2:
████ 9% (21B / 236B)

DeepSeek-V3:
██ 5.5% (37B / 671B)

GPT-OSS-120B:
██ 4.4% (5.1B / 117B)

Year:    2022    2023    2024    2025
Trend:   Coarse  ────────────────► Fine-grained
Sparsity: Low  ────────────────► High
```

## 9. Layer architecture variations

```mermaid
graph TB
    subgraph Llama["Llama (Dense)"]
        L1["Embedding"] --> L2["Attention"]
        L2 --> L3["MLP dense"]
        L3 --> L4["Norm"]
        L4 -.->|"x32 layers"| L2
        L4 --> L5["LM head"]
    end

    subgraph Mixtral["Mixtral"]
        M1["Embedding"] --> M2["Attention"]
        M2 --> M3["MoE block<br/>8 experts, top-2"]
        M3 --> M4["Norm"]
        M4 -.->|"x32 layers"| M2
        M4 --> M5["LM head"]
        style M3 fill:#cfc
    end

    subgraph DeepSeek["DeepSeek-V3"]
        D1["Embedding"] --> D2["Attention MLA"]
        D2 --> D3{"layer under 3?"}
        D3 -->|"Yes"| D4["MLP dense"]
        D3 -->|"No"| D5["MoE block<br/>256 + 1 shared, top-8"]
        D4 --> D6["Norm"]
        D5 --> D6
        D6 -.->|"x61 layers"| D2
        D6 --> D7["LM head"]
        style D5 fill:#cfc
        style D3 fill:#ffc
    end

    subgraph Jamba["Jamba"]
        J1["Embedding"] --> J2{"Layer type?"}
        J2 -->|"attn"| J3["Attention"]
        J2 -->|"mamba"| J4["Mamba SSM"]
        J3 --> J5{"ffn type?"}
        J4 --> J5
        J5 -->|"dense (50%)"| J6["MLP"]
        J5 -->|"sparse (50%)"| J7["MoE block<br/>16 experts, top-2"]
        J6 --> J8["Norm"]
        J7 --> J8
        J8 -.->|"x32 layers"| J2
        J8 --> J9["LM head"]
        style J7 fill:#cfc
        style J4 fill:#fcc
        style J3 fill:#cff
    end
```

**Caption**: Layer pattern comparison. Llama mọi layer dense. Mixtral mọi layer MoE. DeepSeek-V3 3 layer đầu dense, rest MoE. Jamba mixed (Mamba/Attention × Dense/MoE).

## 10. KV cache size comparison

```
KV Cache Memory at 128k Context (batch=1, bf16):

Llama-3-70B (MHA, 8 KV heads):
██████████████████████████████ 41 GB

DeepSeek-V3 (MLA, 512 latent):
███████ 8 GB (5x smaller!)

Mixtral 8x7B (GQA, 8 KV heads):
███████████ 17 GB

GPT-OSS-120B (GQA, 8 KV heads):
█████████ 12 GB
```

## 11. Communication bandwidth tier

```
Bandwidth tier per all-to-all (DeepSeek-V3 forward, N=4096):

Single GPU (no comm):  ~0 ms
██

NVLink (intra-node, 900 GB/s):  ~60 ms
████████████████████

InfiniBand 400G (cross-node, 50 GB/s):  ~1090 ms
██████████████████████████████████████████████████████████████████████

InfiniBand 200G:  ~2180 ms
████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████
```

**Caption**: All-to-all latency tier. NVLink (single-node) practical cho EP. Cross-node IB nghiêm trọng bottleneck.

## 12. Routing entropy evolution

```
Routing entropy H(p) during training (8-expert model):

H (nats)
log(8)=2.08 |█  ← Init (uniform)
       1.80 | ███
       1.50 |    █████████
       1.20 |             █████████  ← Healthy (specialization)
       0.90 |                       ████████
       0.60 |                                ███████  ← Stable
       0.30 |
        0.0 |__________________________________________
            0    20k   50k   100k   200k   500k   step

UNHEALTHY (no aux loss):
H (nats)
log(8)=2.08 |█  ← Init
       1.50 |  ███
       1.00 |     █████
       0.50 |          ██████
       0.10 |                ███████████████████  ← Collapse (1 expert dominates)
        0.0 |__________________________________________
```

**Caption**: Healthy training: entropy stabilize ~1.0-1.5 (chuyên hoá nhưng không collapse). Unhealthy: entropy → 0 (expert collapse).

## 13. FLOPs breakdown (Mixtral 8x7B prefill)

```
FLOPs breakdown per forward (Mixtral 8x7B, prefill N=4096):

Attention (Q,K,V,O + softmax): 1.0 TFLOP
██████████ 11%

Router: 0.01 TFLOP (negligible)
▏ <1%

Expert (top-2 of 8): 7.8 TFLOPs
█████████████████████████████████████████████████████████████████████████████ 79%

Layer norm: 0.05 TFLOP
▏ 1%

Other (residual, etc.): 1.0 TFLOP
██████████ 10%

Total: ~10 TFLOPs per forward
```

**Caption**: Expert compute dominate (79%). Router cost negligible. Attention significant but ~11%.

## 14. Decision tree: MoE vs Dense

```mermaid
flowchart TD
    Start["Need a model"] --> Q1{"Total params target?"}
    Q1 -->|"<= 8B"| Dense1["Use Dense<br/>Llama-3-8B, Mistral-7B"]
    Q1 -->|"8-30B"| Q2{"Latency p50 under 50ms?"}
    Q2 -->|"Yes"| Dense2["Use Dense<br/>Mistral-7B, Llama-3-13B"]
    Q2 -->|No| MoESmall["Use MoE coarse<br/>Mixtral 8x7B, PhiMoE"]
    Q1 -->|"30-100B"| Q3{"Hardware?"}
    Q3 -->|"Single GPU 80GB"| Quant["Quantize<br/>Mixtral 4-bit"]
    Q3 -->|"4-8 GPU"| TPMoE["TP MoE<br/>Mixtral 8x7B, Qwen3-30B"]
    Q3 -->|"Multi-node"| EPMoE["EP + TP<br/>Mixtral 8x22B, DeepSeek-V2"]
    Q1 -->|">= 100B"| Q4{"Need SOTA?"}
    Q4 -->|"Yes"| Frontier["Fine-grained MoE<br/>DeepSeek-V3, GPT-OSS-120B"]
    Q4 -->|"No"| Quant2["Quantized big dense<br/>Llama-3-70B 4-bit"]

    style Dense1 fill:#ccf
    style Dense2 fill:#ccf
    style MoESmall fill:#cfc
    style Quant fill:#fcf
    style TPMoE fill:#cfc
    style EPMoE fill:#cfc
    style Frontier fill:#fac
    style Quant2 fill:#ccf
```

**Caption**: Decision tree khi chọn architecture. Tham khảo Phần 5 Chương 5 cho details.

## 15. Throughput chart by configuration

```
TPS (tokens/sec) by configuration (DeepSeek-V3, prefill N=4096):

1× H100, single-batch decode:
██████ 45 TPS (memory-bandwidth limited)

8× H100 NVLink, batch 8, decode:
█████████████████████████████ 600 TPS (utilization improves)

8× H100 NVLink, batch 32, mixed prefill+decode:
████████████████████████████████████████████████ 1500 TPS

4× H100 × 2 nodes IB 400G, batch 32:
███████████████ 350 TPS (cross-node bottleneck)
```

**Caption**: Throughput scales với batch + intra-node bandwidth. Cross-node communication penalty đáng kể.

## Đề xuất sử dụng

Các diagram trên có thể dùng:

1. **Slide presentation**: copy-paste Mermaid blocks vào draw.io hoặc Mermaid Live Editor để export PNG/SVG.
2. **Blog post**: render trực tiếp trong Markdown.
3. **Tài liệu cho team**: kèm theo derivation Chương 3 cho training discussion.
4. **Quick reference**: print Chương 6 ra giấy.

Để export PNG từ Mermaid:

```bash
npm install -g @mermaid-js/mermaid-cli
mmdc -i diagram.mmd -o diagram.png -t dark
```

Phần 6 kết thúc. Toàn bộ chuỗi bài giảng hoàn chỉnh.
