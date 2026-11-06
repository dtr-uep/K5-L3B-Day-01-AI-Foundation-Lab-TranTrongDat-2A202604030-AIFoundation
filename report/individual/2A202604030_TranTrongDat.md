# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202604030
- **Họ và tên:** Trần Trọng Đạt
- **SEED** (in ở ô đầu notebook): 7883

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | V |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | V |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | V |

## 2. Kết quả chính

*Chép bảng so sánh cuối Phần 5 (F1 test, F1 SUB 40, p95 ms/tin, USD/1k tin) của chính bạn.*
                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.988   0.792      0.879        0.889   
2. BoW + MLP (deep learning)         0.958   0.901      0.929        0.947   
3a. Encoder (MiniLM) + kNN           0.890   0.960      0.924        1.000   
3b. Encoder (MiniLM) + LogReg        0.941   0.950      0.946        0.923   
3c. Encoder (MiniLM) + MLP head      0.942   0.960      0.951        0.950   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            6.202       8870.248       0.060   
2. BoW + MLP (deep learning)        12.100      28268.622       0.224   
3a. Encoder (MiniLM) + kNN         202.729          9.558     113.943   
3b. Encoder (MiniLM) + LogReg      203.949         10.131      55.686   
3c. Encoder (MiniLM) + MLP head    206.678          9.850      59.879   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.265           0.0  
2. BoW + MLP (deep learning)          0.268           0.0  
3a. Encoder (MiniLM) + kNN          377.270           0.0  
3b. Encoder (MiniLM) + LogReg       125.734           0.0  
3c. Encoder (MiniLM) + MLP head     147.012           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.

## 3. Dự đoán và thực tế

*Chọn 3 dự đoán `predict(...)` lệch nhiều nhất so với kết quả. Mỗi cái: bạn đoán gì, ra gì, vì sao lệch.*
Dự đoán thay đổi đường loss trên learning rate, dự đoán là sẽ khó converge và giao động ở các epoch, thực tế chỉ giao động ở 2 epoch đầu và vẫn converge rất sớm.

Dự đoán bỏ ReLU thì model sẽ ra kết quả kém hơn MLP nhưng f1 kết quả tốt hơn. MLP không hoàn toàn đúng trong mọi trường hợp, logistic regression với learning rate có kiểm soát có thể tốt hơn.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
