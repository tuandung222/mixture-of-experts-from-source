---
title: Tổng quan Phần 5
---

# Phần 5: Design comparison và decision guide

Phần cuối. Chuyển từ "đọc model cụ thể" sang "so sánh ngang và đưa ra quyết định". Có 4 chương:

- Chương 2: **Routing comparison matrix**. Bảng matrix router trên 10 model.
- Chương 3: **Load balancing comparison**. So sánh aux loss, z-loss, bias adjustment.
- Chương 4: **Expert design comparison**. Coarse vs fine-grained, shared, FFN ratio.
- Chương 5: **Khi nào dùng MoE vs dense**. Decision tree thực hành.

## Mục tiêu Phần 5

Sau Phần 5, bạn:

1. Có bảng tra cứu nhanh thiết kế MoE.
2. Biết trade-off mỗi design choice.
3. Có decision tree để chọn paradigm cho task riêng.
4. Hiểu xu hướng tương lai (fine-grained, sigmoid + bias adjust, dropless).

## Tại sao tách Phần 5

Phần 3 và Phần 4 dài (12 + 6 chương). Đọc xong, bạn có nhiều thông tin nhưng không có "30-second answer" cho câu hỏi "Should I use MoE?".

Phần 5 là **synthesis**. Mỗi chương trả lời một câu hỏi cụ thể:

- "Router nào tốt cho task của tôi?"
- "Aux loss coef bao nhiêu?"
- "Bao nhiêu expert là vừa?"
- "MoE vs dense, chọn gì?"

Nếu bạn skim từ đầu, có thể chỉ đọc Phần 5 để có overview, rồi quay lại Phần 3 model cụ thể nếu cần.

Chương sau bắt đầu với routing matrix.
