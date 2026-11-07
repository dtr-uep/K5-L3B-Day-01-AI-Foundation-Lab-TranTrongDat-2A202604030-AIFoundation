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

Dự đoán ở phần stress test: dự đoán loại "Tưởng spam nhưng không phải" sẽ khiến BoW gặp khó khăn nhất nhưng hầu hết model đều làm tốt ở đây. Kỳ vọng là các từ spam sẽ nặng cho trọng số để dự đoán là ham nhưng có lẽ vì có nhiều từ ngữ thông thường nên BoW vẫn có thể đoán đúng.

## 4. Phân tích 50 case khó

*Model nào tụt nhiều nhất, ở nhóm case nào? Nêu 2 case cụ thể và cơ chế khiến model sai.*
Bị lỗi ở 2 model LLM ở một số phần, chủ yếu trả về None.
Model tụt nhiều nhất không tính LLM về tổng thể là 2 model BoW. Tụt nhiều nhất ở phần Thao túng / Lừa Đảo, Spam giọng tự nhiên và đặc biệt là né từ khóa. Feature BoW phụ thuộc hoàn toàn vào sự xuất hiện của từ khóa đồng thời lại nhạy cảm với từ ngữ thông thường do thiên kiến về ham, từ đó dẫn đến model dễ bị sai ở các case mà các từ khóa cần thiết không có hoặc không đủ để dự đoán.

## 5. Thử thách cá nhân

*Đề theo SEED của bạn, cách làm, kết quả, giải thích.*
SEED = 7883
Cách làm:  
Prompt sửa requirement thành yêu cầu định dạng JSON với text (yêu cầu giống như chỉ label) và reason.
Sử dụng regex để trích xuất phần trong ngoặc có nhóm pattern là "value":"key", sau đó dùng json.loads sau khi loại bỏ \n để trích dưới dạng json.

Kết quả:
Cho thấy chi phí tăng gấp 2 lần và độ trễ tăng 2s.

## 6. Phương án cho sản phẩm thật

*Với yêu cầu p95 < 50 ms/tin cho 10 triệu SMS/ngày: chọn phương án nào, dẫn số liệu nào, chấp nhận đánh đổi gì.*
Phương án chọn mô hình encoder (MiniLM) + LogReg có độ trễ phù hợp và loss thấp nhất, đồng thời trong stress test có phần thể hiện tương đương với các model khác. Chấp nhận sử dụng thêm token cho model Embedding nhưng chi phí vẫn rẻ hơn nhiều so với LLM.

## 7. AI log và điều đã học

*Mỗi lần dùng AI: hỏi gì, nhận gì, bạn sửa gì. Sau đó 2–3 câu: điều bạn hiểu ra hôm nay mà trước đó chưa hiểu.*
Đã sử dụng AI để tra cách dùng regex trích xuất JSON từ string text ở phần challenge. AI trả lời: re.search(r'{[^{}]*}', s) và đã copy nguyên cái đó thêm phần trích lấy nội dung từ match và xử lý None.

Đã sử dụng AI để tra cách sử dụng time.perf_counter trong phần viết chạy llm theo batc. AI trả về full code mẫu, chỉ lấy phần sử dụng time.perf_count để cho vào code của bản thân.

Đã sử dụng AI để tra câu code parser_label dựa trên nguyên đề bài, AI trả về full script đoạn parser đó, bản thân không sửa gì hơn.

Hiện tại các regex pattern của AI trả về cũng chưa đủ để bao phủ tất cả các case, hiểu thêm rằng cần xử lý cả format đầu ra mà không hoàn toàn phụ thuộc vào AI bên cạnh các cách xử lý về bảo mật và nội dung.
