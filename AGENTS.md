# AGENTS.md

Hướng dẫn cho AI agent (Cascade, Cursor, Aider, Codex, ...) làm việc trên repo này. Đọc trước khi sửa bất kỳ file nào.

## 1. Project context

Repo là một **chuỗi bài giảng tiếng Việt** (Docusaurus 3) phân tích 10 model Mixture of Experts trong HuggingFace `transformers`: Mixtral, Switch Transformers, DeepSeek-V3, Qwen3-MoE, GPT-OSS (5 core) + OLMoE, JetMoE, Jamba, NLLB-MoE, PhiMoE (5 supplementary). Audience: ML engineer đã quen Transformer dense (không bắt buộc quen MoE), muốn hiểu sâu về router, expert dispatch, load balancing, expert parallelism, quantization. Triết lý: bottom-up, neo mọi khái niệm vào file/class/dòng code cụ thể trong codebase transformers thật.

**Codebase tham chiếu (source of truth)**: `/Users/admin/TuanDung/repos/transformers` (HF transformers checkout, tag tại thời điểm viết). Mọi citation phải khớp class/file thực tồn tại ở đây.

**Repo liên quan**: `transformers-internals-foundation` (https://github.com/tuandung222/transformers-internals-foundation) cùng tác giả đã cover attention, KV cache, generate, conventions. Khi nhắc đến những chủ đề này, có thể cross-reference (đường link đầy đủ) thay vì giải thích lại.

## 2. Hard rules (phải tuân thủ, có CI/QA enforce)

- **`README.md` phải rỗng** (0 bytes). Lý do: privacy. Check bởi `scripts/qa_docs.py`.
- **Không em-dash (ký tự U+2014)** trong bất kỳ file nào ngoài `README.md`. Dùng dấu phẩy, dấu hai chấm, hoặc parenthesis. Check bởi `qa_docs.py`.
- **Không leak identity cá nhân**: pattern `Small-?Qwen|CLIP-?HAR|LLMs?-with-Semantic-Search|Open-?vocabulary-Action-Recognition|tuandung222` không được xuất hiện trong `docs/`. Check bởi `qa_docs.py`.
- **Privacy infra**: `static/robots.txt` phải `User-agent: *\nDisallow: /`. `docusaurus.config.ts` phải có meta `noindex,nofollow,noarchive,nosnippet` và `sitemap: false`. Đừng đụng vào.
- **Sidebar ID phải khớp file slug**. Mỗi `id` trong `sidebars.ts` tương ứng với `docs/<id>.md` hoặc `docs/<id>/index.md`. Check bởi `qa_docs.py`.
- **Link nội bộ tuyệt đối `](/docs/...)` phải trỏ tới file tồn tại**. Check bởi `qa_docs.py`.
- **Chỉ commit khi `npm run verify` pass** (gộp qa_docs + typecheck + build).

## 3. Writing conventions

**Ngôn ngữ**: Tiếng Việt cho prose. Giữ thuật ngữ tiếng Anh khi: tên class/method (`PreTrainedModel`, `from_pretrained`), tên kỹ thuật phổ biến (attention, cache, embedding, tokenizer, decoder, prefill, decode), tên paper.

**Tone**: Trực tiếp, không filler. Không mở đầu bằng "Trong chương này chúng ta sẽ...". Vào thẳng nội dung. Không xưng "chúng tôi" hoa mỹ; dùng "ta" hoặc giọng impersonal khi cần.

**Bottom-up**: Luôn dẫn code/snippet/số liệu trước, giải thích trừu tượng sau. Không tổng quát hoá khi chưa có một ví dụ cụ thể.

**Không emoji** trong docs (trừ khi user yêu cầu).

**Structure mặc định mỗi chapter**:

1. Frontmatter `title:`.
2. `# Heading` trùng title.
3. 1 đoạn mở đầu nêu vì sao chapter tồn tại (1-3 câu).
4. Section chính với `## H2`.
5. Snippet code (Python hoặc pseudocode) đi kèm giải thích.
6. Số liệu cụ thể nếu có (memory, FLOPs, latency).
7. Section `## Pitfall` hoặc `## Pitfalls` ở cuối khi có nhiều case fail thường gặp.
8. Câu cuối trỏ tới chapter kế ("Chương sau ta...").

**Citation source code**: Khi tham chiếu, ghi tên file + class + (tuỳ chọn) dòng. Ví dụ: `src/transformers/modeling_utils.py`, `class PreTrainedModel`. Không bịa dòng số nếu chưa verify.

**Số liệu**: Khi nêu memory/latency, ghi rõ model + batch + context. Ví dụ: "Llama-3 8B, batch=1, context 128k: 17.18 GB". Tránh hand-wave kiểu "memory rất lớn".

## 4. MDX gotchas (lessons từ session đầu tiên, painful)

Docusaurus 3 dùng MDX 3. MDX parse `{...}` thành JSX expression **ngay cả trong khối toán `$$...$$`**. Hậu quả:

- `$$H_{kv}$$` → MDX coi `{kv}` là JSX, raise `ReferenceError: kv is not defined` lúc build SSR. **Fix**: dùng `\text{kv}` (`$$H_{\text{kv}}$$`), hoặc bỏ math, dùng code fence ASCII: `H_kv` trong ` ``` ` block.
- `\text{cache}` thì OK vì MDX không parse khi có backslash macro phía trước. An toàn nhất: mọi subscript identifier dùng `\text{...}`.
- `<1%` ngoài code block → MDX coi `<1` là JSX tag mở, raise "Unexpected character `1` before name". **Fix**: viết "dưới 1%" hoặc bọc backtick `` `<1%` ``.
- `<->` ngoài code block → cùng vấn đề. **Fix**: backtick `` `<->` ``.
- Curly literal trong prose `{foo}` cũng có rủi ro. Bọc backtick nếu là code, hoặc escape `\{foo\}`.

**Quy tắc**: chạy `npm run build` mỗi khi thêm math hoặc dấu `<`. `qa_docs.py` và `tsc` không bắt được lỗi MDX.

## 4b. Mermaid gotchas (lỗi không bị bắt bởi docusaurus build, painful)

Docusaurus theme-mermaid **embed Mermaid source vào JS bundle và render client-side**. `npm run build` vẫn PASS dù syntax sai, user chỉ thấy lỗi "Parse error on line N" trong browser console khi mở page. Vì vậy **phải validate Mermaid trước commit** bằng `npm run qa:mermaid` (gọi `scripts/validate_mermaid.mjs`, parse từng block qua `mermaid.parse()`).

Các pattern gây lỗi đã gặp:

- **Nested square brackets**: `SX[z=[3, 1, -1]]` → parser tưởng `[` thứ hai là node shape lồng. **Fix**: bọc toàn bộ label trong dấu nháy → `SX["z = (3, 1, -1)"]`, hoặc đổi `[]` thành `()` trong text.
- **Bare ellipsis**: `EE[...]` và `DE3[...8 experts...]` → mở đầu bằng `...` parser confuse. **Fix**: quote `EE["..."]`.
- **Pipe label trần `|+|` và `|-|`**: `-` là char reserved cho arrow `-->`. Parser tưởng arrow tới ngắn. **Fix**: quote `|"positive"|` và `|"negative"|` (hoặc đổi nội dung).
- **Unquoted `<` trong diamond**: `Q2{Latency p50 < 50ms?}` → có nguy cơ, `<` có thể bị confuse. **Fix**: quote `Q2{"Latency p50 under 50ms?"}` hoặc đổi sang "under".
- **Unicode arrow `→`, `≤`, `≥`, `×` trong labels**: Mermaid version 11 đã khá tốt với unicode nhưng còn có edge case. **Fix**: thay bằng ASCII (`to`, `<=`, `>=`, `x`), hoặc quote.
- **HTML `<br/>` trong unquoted bracket**: thường OK (Mermaid xử lý như HTML markup), nhưng safer khi quote.
- **Apostrophe `'`**: `Can't fit` có thể confuse parser. **Fix**: đổi `Cannot fit` hoặc quote.

**Quy tắc vàng**: mọi node label trong Mermaid (cả `[...]`, `{...}`, `(...)`) và mọi pipe label (`|...|`) **nên luôn bọc trong dấu nháy kép** nếu có bất kỳ special char (space + punctuation, brackets, comparison ops, arrows, math symbols). Đừng "tiết kiệm" bỏ quote.

`scripts/validate_mermaid.mjs` dùng `mermaid.parse()` qua jsdom để catch all errors offline. Đã tích hợp vào `npm run verify` (bước `qa:mermaid`).

## 5. Workflow chuẩn

Trước khi commit bất kỳ thay đổi nào trong `docs/`:

```bash
npm run verify
# = qa_docs.py && qa:mermaid && tsc && docusaurus build
```

Nếu chỉ sửa file nhỏ, có thể chạy từng bước:

```bash
python3 scripts/qa_docs.py     # nhanh, bắt em-dash, sidebar, README
node scripts/validate_mermaid.mjs   # bắt lỗi Mermaid parse (xem mục 4b)
npm run typecheck              # bắt lỗi TS trong sidebars.ts, docusaurus.config.ts
npm run build                  # CRITICAL: bắt lỗi MDX (xem mục 4)
```

`npm run start` chỉ cho dev preview, **không phát hiện hết** lỗi build-time. Phải `build`. Build CHỈ bắt được MDX-level error; Mermaid parse error **không bị bắt** bởi build, phải `qa:mermaid` riêng.

## 6. Cấu trúc repo

```
docs/
  intro.md                    # Trang giới thiệu
  00-tong-quan/               # Part 0: orientation (4 chương)
  01-foundations/             # Part 1: router, routing, balancing, capacity, shared (6)
  02-hf-moe-infra/            # Part 2: integrations/moe.py walkthrough (5)
  03-models/                  # Part 3: 10 model walkthroughs (12)
  04-cross-cutting/           # Part 4: EP, TP, quant, serving, training (6)
  05-comparison/              # Part 5: design comparison & decision guide (5)
  06-mathematical-modeling/   # Part 6: math derivations, FLOPs, diagrams (6)
  resources/
    glossary.md
    cheatsheet.md
    references.md
sidebars.ts                   # Phải đồng bộ với docs/ tree
docusaurus.config.ts          # baseUrl, organizationName, math plugin
scripts/
  qa_docs.py                  # QA gate (em-dash, sidebar, identity leak)
  validate_mermaid.mjs        # QA gate cho Mermaid parse (mục 4b)
.github/workflows/deploy.yml  # GitHub Pages auto-deploy on push main
```

**Naming**: file chapter dạng `NN-slug-tieng-viet.md` (NN = 01-99, slug snake-case không dấu). Khi đổi tên file, **phải** update `sidebars.ts` đồng thời.

**Thêm Part mới**: tạo thư mục `06-...`, viết `01-overview.md`, thêm category block vào `sidebars.ts`, cập nhật `docs/00-tong-quan/04-roadmap.md`.

## 7. Patterns đã thiết lập

- **Overview chapter** mỗi part: nêu mục tiêu, list các chương, đặt bối cảnh trong toàn series.
- **Walkthrough chapter**: đi qua một class/function thực, copy snippet, annotate dòng.
- **Comparison chapter**: bảng so sánh backend/strategy/variant.
- **Pitfall section** cuối chapter dài: liệt kê 3-5 case fail thường gặp với fix.
- **Glossary entry**: 1-2 câu định nghĩa + chỉ chapter chi tiết.
- **Cheatsheet**: snippet copy-paste-ready, không giải thích dài.

## 8. Source-of-truth references

Khi cần verify code transformers cho MoE:

```bash
# Codebase HF transformers local
/Users/admin/TuanDung/repos/transformers/src/transformers/
  integrations/moe.py                              # ExpertsInterface, batched_mm, grouped_mm
  integrations/mxfp4.py                            # MXFP4 quantization for MoE
  integrations/finegrained_fp8.py                  # FP8 quantization for MoE
  models/mixtral/modeling_mixtral.py               # Baseline reference (~704 dòng)
  models/switch_transformers/modeling_switch_transformers.py  # Encoder-decoder + top-1
  models/deepseek_v3/modeling_deepseek_v3.py       # State-of-the-art aux-free + shared
  models/qwen3_moe/modeling_qwen3_moe.py           # Modern infra với grouped_mm
  models/gpt_oss/modeling_gpt_oss.py               # Production với MXFP4
  models/olmoe/modeling_olmoe.py                   # Open recipe
  models/jetmoe/modeling_jetmoe.py                 # MoA + MoE
  models/jamba/modeling_jamba.py                   # Mamba + MoE hybrid
  models/nllb_moe/modeling_nllb_moe.py             # Translation, expert dropout
  models/phimoe/modeling_phimoe.py                 # Small-scale MoE
```

Tool agent dùng: `grep_search` cho targeted search, `code_search` cho exploratory với câu hỏi cụ thể, `read_file` để verify dòng số.

## 9. Deployment

- Repo: `tuandung222/mixture-of-experts-from-source` trên GitHub.
- Pages URL: https://tuandung222.github.io/mixture-of-experts-from-source/
- Auto-deploy: `.github/workflows/deploy.yml` chạy on push `main`. Verify build local trước khi push (workflow sẽ fail nếu build fail).
- Config `docusaurus.config.ts` hard-code `organizationName: tuandung222`, `baseUrl: /mixture-of-experts-from-source/`. Đừng đổi trừ khi migrate account.

## 10. Anti-patterns đã gặp (avoid)

- **Tạo file mới khi chỉ cần edit**. Repo này đã có cấu trúc, đừng thêm scratch file (`notes.md`, `tmp.md`).
- **Liệt kê inline bullet** thay vì list block. Markdown list phải có endline trước.
- **Em-dash trong văn bản** (style preference + qa enforce).
- **Mock số liệu không reasonable**. Ví dụ memory 10x sai. Nếu không chắc, ghi "tham khảo" hoặc bỏ.
- **Hand-wave reference**: "transformers có class abc xử lý việc này". Phải ghi chính xác tên class và file.
- **Generic intro section**: "Cross-attention là một kỹ thuật quan trọng...". Bỏ. Vào thẳng code.
- **Math block với `{identifier}` không bọc `\text`**: gây MDX fail (xem mục 4).
- **Đụng vào `README.md`**: phải giữ rỗng.
- **Sửa modeling file của HF transformers**: repo này chỉ documentation, không fork transformers.

## 11. Quick checklist khi thêm 1 chương

- [ ] File đặt đúng folder, naming `NN-slug.md`.
- [ ] Frontmatter `title:` set.
- [ ] Heading 1 trùng title.
- [ ] Không em-dash.
- [ ] Math block với identifier dùng `\text{}` hoặc bỏ math.
- [ ] Mọi `<digit` và `<-` bọc backtick.
- [ ] Mermaid: mọi node/pipe label có special char được quote (mục 4b).
- [ ] Citation code có file path + class name thực.
- [ ] Số liệu có context (model + batch + seq_len).
- [ ] Pitfall section nếu chapter dài (>200 dòng).
- [ ] Câu cuối trỏ chapter kế.
- [ ] `sidebars.ts` có entry cho file mới.
- [ ] `npm run verify` pass (gồm qa:mermaid).
- [ ] Commit message ngắn gọn, mô tả thay đổi.
