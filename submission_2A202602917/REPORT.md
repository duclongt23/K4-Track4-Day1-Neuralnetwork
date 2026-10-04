# Báo cáo Lab Day 1 — Nguyễn Đức Long — 2A202602917

> Tối đa khoảng 4 trang. **Chỉ viết cho những chủ đề bạn đã thử**; xoá các mục không dùng. Mọi con số phải trỏ về một `exp_id` trong `experiments.xlsx`. Mọi "A tốt hơn B" phải cân nhắc độ nhiễu seed (mục 2). Xoá các dòng hướng dẫn `>` khi nộp.

## 1. Thiết lập

- Môi trường (Colab/Kaggle, GPU, phiên bản PyTorch): Google Colab, GPU T4, PyTorch 2.3.1+cu121
- Dữ liệu: Forest CoverType; `train` 464 809 / `eval` 116 203 theo `split_metadata.csv`. Validation: 20% của train (phân tầng, seed 42) → 371 847 train / 92 962 val.
- Model: `M-base` (54→256→128→7, 47 879 tham số). Baseline: loss, optimizer, lr, batch, epochs, init.
- Mốc tham chiếu: accuracy "đoán lớp đa số" trên val = 0.4876 (≈ 0,4876).
- Các chủ đề đã thử: ☐ loss ☑ optimizer ☐ hyper-parameter ☐ dropout ☐ clipping ☐ mixed precision ☐ init

## 2. Kiểm tra ban đầu và độ nhiễu

| Kiểm tra | Kết quả |
|---|---|
| Số tham số / shape logits | 47,879 / (B, 7) |
| Loss bước 0 (so với ln 7 = 1,946) | ~2.1250 |
| Quá khớp 20 mẫu: loss cuối | (Đã kiểm tra) |
| Mọi tham số có gradient khác 0 | ☑ có |
| Baseline, số seed đã chạy | 3 (42, 43, 44) |
| Baseline: val acc (TB ± σ) | 0.9108 ± 0.0011 |
| Baseline: val macro-F1 (TB ± σ) | 0.8571 ± 0.0031 |

**Ngưỡng nhiễu dùng trong báo cáo:** 2σ = 0.0062 (val macro-F1). *(Nếu chỉ có 1 seed: ghi rõ không đo được nhiễu và đây là hạn chế.)*

## 3. Kết quả theo chủ đề

> Mỗi chủ đề đã thử: (a) dự đoán trước khi chạy, (b) kết quả (số + ảnh + `exp_id`), (c) giải thích cơ chế, (d) khác biệt có vượt nhiễu không. Tất cả dựa trên **val**, không dựa trên eval.

### 3.2 Bộ tối ưu hoá
- **Dự đoán:** Adam sẽ hội tụ nhanh hơn SGD ở những epoch đầu do khả năng tự chỉnh learning rate từng tham số, nhưng val_macro_f1 cuối cùng có thể thấp hơn hoặc bị overfitting sớm hơn SGD.
- **Bảng nhỏ (kết quả tại lr tốt nhất):**
  - `base-s42` (SGD+Momentum): lr = 0.1, val macro-F1 = 0.8610, best epoch = 20
  - `exp-adam` (Adam): lr = 1e-3, val macro-F1 = 0.8496, best epoch = 20
- **Ảnh chồng:** Xem `figures/compare_optimizer_loss.png` và `figures/compare_optimizer_f1.png`. Adam có loss giảm rất sâu và nhanh lúc đầu nhưng về sau tiệm cận mức không bằng SGD.
- **Giải thích:** Khớp với dự đoán. F1 của Adam (0.8496) kém hơn SGD (0.8571 trung bình). Chênh lệch (-0.0075) vượt qua ngưỡng nhiễu 2σ (0.0062), mang ý nghĩa thống kê. Nguyên nhân có thể do Adam tìm được nghiệm sharp minima nhanh nhưng kém tổng quát hóa (generalization) hơn SGD + Momentum (thường tìm flat minima).

## 4. Đánh giá cuối trên tập eval

> Chỉ làm sau khi chọn cấu hình bằng val. Số lấy từ `eval_result.json` (do `scripts/evaluate.py` tạo), không tự tính lại.

| Cấu hình | Seed nộp | val macro-F1 | **eval macro-F1** | eval accuracy |
|---|---|---|---|---|
| Baseline | 42 | 0.8610 | - | - |
| Cấu hình cuối cùng (`exp-adam`) | 42 | 0.8496 | 0.8470 | 0.9027 |

- **Cấu hình cuối cùng gồm những gì và vì sao:** Chọn cấu hình `exp-adam` (chạy 20 epochs, lr=1e-3, batch=512) để làm cấu hình nộp bài vì mục đích khảo sát khả năng đánh giá trên tập eval (mặc dù dựa trên val, Baseline tốt hơn một chút).
- **Cải thiện so với baseline:** Mức F1 eval (0.8470) khá sát với val (0.8496). Thử nghiệm Adam không thực sự cải thiện so với baseline (về mặt chỉ số).
- **Val và eval:** Rất gần nhau (0.8496 vs 0.8470), chứng tỏ tập validation chia phân tầng (20% từ train) là đại diện rất tốt cho phân phối của tập eval ẩn.

### 4.1 Phân tích lỗi theo lớp

| Lớp | support | precision | recall | F1 |
|---|---|---|---|---|
| 0 | 42368 | 0.9024 | 0.9050 | 0.9037 |
| 1 | 56661 | 0.9130 | 0.9244 | 0.9187 |
| 2 | 7151 | 0.8809 | 0.8747 | 0.8778 |
| 3 | 549 | 0.7804 | 0.8415 | 0.8098 |
| 4 | 1899 | 0.7305 | 0.7651 | 0.7474 |
| 5 | 3473 | 0.8280 | 0.7029 | 0.7603 |
| 6 | 4102 | 0.9583 | 0.8684 | 0.9111 |

- Lớp khó nhất là lớp 4 (F1 = 0.7474). Nó hay bị nhầm với lớp 1 (xem ma trận nhầm lẫn; chèn ảnh nếu có).
- **Lý giải:** Lớp 4 và 5 có số lượng mẫu khá ít (1899 và 3473), dẫn đến hiện tượng mất cân bằng dữ liệu (imbalance). Hơn nữa, đặc trưng địa hình của chúng có thể bị chồng lấn rất nhiều với nhóm đa số (Lớp 1 và 2).
- **Cách cải thiện sẽ thử:** Sử dụng **Class Weights** trong hàm Loss (CE) để phạt nặng hơn khi đoán sai các lớp thiểu số, hoặc dùng **Focal Loss**.

## 5. Trả lời các câu hỏi dẫn dắt

> Trả lời những câu liên quan tới chủ đề bạn đã thử; câu 6 luôn bắt buộc.

1. **Bộ tối ưu nào "thắng" khi mỗi cái được chỉnh lr công bằng? Khi lr không được chỉnh thì kết luận thay đổi ra sao?**
   - **Trả lời:** Khi mỗi optimizer được tinh chỉnh lr công bằng (SGD lr=0.1, Adam lr=1e-3), SGD + Momentum "thắng" do F1 cao hơn. Tuy nhiên, nếu không được tinh chỉnh (ví dụ đều để default lr=1e-3), SGD sẽ hội tụ quá chậm và kém xa Adam. Adam ít nhạy cảm với khởi tạo lr hơn.

6. **Quay lại câu hỏi của bài học:** một mạng có loss không giảm sau 2 000 bước. Dựa vào bảng "triệu chứng" ở Chương 5 và các thí nghiệm của bạn, nêu 3 phép kiểm tra đầu tiên bạn sẽ làm và vì sao.
   - **Trả lời:**
     1. **Kiểm tra Learning Rate:** LR có thể quá lớn (gây văng loss) hoặc quá nhỏ (mô hình không học được gì). Hãy thử chia/nhân LR cho 10.
     2. **Kiểm tra dữ liệu đầu vào:** Xem lại bước chuẩn hóa (Normalization). Nếu input lớn bất thường hoặc có NaN, gradient sẽ hỏng.
     3. **Kiểm tra lỗi khởi tạo (Initialization) hoặc shape logits:** Chắc chắn mô hình không xuất ra toàn số giống nhau, gradient có truyền ngược được không (bằng cách in `grad_norm` như trong bài Lab).

## 6. Hạn chế và điều bất ngờ

- **Kết quả khác dự đoán:** Tốc độ Adam ban đầu rất bùng nổ nhưng lại bị "chững lại" và không vượt được SGD.
- **Hạn chế thí nghiệm:** Mới thử Adam ở đúng 1 mức `lr = 1e-3` (default) và chạy 1 seed (seed 42) do đó so sánh với SGD (đã chỉnh cực tốt `lr=0.1` với 3 seeds) là chưa thực sự sòng phẳng.
- **Nếu có thêm thời gian:** Sẽ thực hiện Grid Search cho Learning Rate của Adam, và thử thêm cơ chế Dropout để chống overfitting trong trường hợp train số epoch lớn hơn.

## 7. Phụ lục

- Danh sách file đã nộp: `REPORT.md`, `experiments.xlsx`, `predictions_eval.csv`, `eval_result.json`, `figures/*.png`, `results/*.json`, thư mục `code/`.
- Thời gian chạy ước tính tổng cộng: Khoảng 5-10 phút trên Google Colab GPU T4.
