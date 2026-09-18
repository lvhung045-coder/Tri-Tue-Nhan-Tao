# KẾ HOẠCH CHI TIẾT: TÁI CẤU TRÚC HỆ THỐNG GÁN NHÃN & HUẤN LUYỆN MÔ HÌNH (5.000 CÂU)

Tài liệu này ghi lại toàn bộ lộ trình, các bước kỹ thuật và danh sách công việc cần triển khai (Checklist) để chuyển đổi hệ thống phân loại cảm xúc cũ (4 nhãn: `negative_1`, `negative_2`, `negative_3`, `positive`) sang hệ thống nhãn mới đa chiều kết hợp giữa **Sắc thái tình cảm (Sentiment Polarity)** và **Cảm xúc cốt lõi (Basic Emotions)** trên tập dữ liệu chuẩn **5.000 câu**.

---

## 1. HỆ THỐNG NHÃN MỤC TIÊU

Hệ thống nhãn mới được cấu thành từ 2 nhóm với tổng cộng **11 nhãn**:

### Nhóm 1: Sắc thái tình cảm (Sentiment Polarity - 5 mức độ)
1. **Rất tích cực (Very Positive)**: Cực kỳ hài lòng, khen ngợi tuyệt đối, phấn khích cao độ.
2. **Tích cực (Positive)**: Hài lòng, tán thành, có cảm tình tốt.
3. **Trung tính (Neutral)**: Câu thông báo khách quan, nêu sự thật, không bộc lộ thiện cảm hay ác cảm.
4. **Tiêu cực (Negative)**: Không vừa ý, chê bai nhẹ, thất vọng ở mức độ vừa phải.
5. **Rất tiêu cực (Very Negative)**: Căm phẫn, phẫn nộ, chỉ trích gay gắt, tẩy chay kịch liệt.

### Nhóm 2: Cảm xúc cốt lõi (Ekman's Basic Emotions - 6 cảm xúc)
6. **Hạnh phúc (Happiness)**: Cảm giác vui sướng, hài lòng và thỏa mãn, giúp con người kết nối với nhau.
7. **Buồn bã (Sadness)**: Trạng thái chán nản hoặc thất vọng khi đối mặt với sự mất mát hay tổn thương.
8. **Sợ hãi (Fear)**: Phản ứng mạnh mẽ khi gặp nguy hiểm, giúp cơ thể chuẩn bị để chiến đấu hoặc bỏ chạy.
9. **Giận dữ (Anger)**: Cảm giác bực bội, khó chịu khi ranh giới bị xâm phạm hoặc gặp bất công.
10. **Ghê tởm (Disgust)**: Phản ứng từ chối hoặc né tránh trước những thứ gây khó chịu, độc hại.
11. **Ngạc nhiên (Surprise)**: Phản ứng chớp nhoáng trước một sự việc bất ngờ xảy ra xung quanh.

---

## 2. PHÂN TÍCH THIẾT KẾ KIẾN TRÚC MÔ HÌNH (CẦN XÁC ĐỊNH)

Khi kết hợp giữa **5 mức độ sắc thái** và **6 cảm xúc**, có 2 hướng thiết kế bài toán:

### Phương án A: Phân loại 11 lớp phẳng (Single-label 11 classes)
* **Cách thức:** Mỗi câu chỉ được gán đúng 1 nhãn duy nhất trong số 11 nhãn trên (`Softmax` với 11 đầu ra).
* **Ưu điểm:** Đơn giản, dễ huấn luyện, giữ nguyên cấu trúc pipeline hiện tại của dự án (`NUM_LABELS = 11`).
* **Nhược điểm:** Dễ gây nhập nhằng nhãn (Ví dụ: một câu *"Tôi quá hạnh phúc"* vừa mang nhãn `Hạnh phúc` vừa mang nhãn `Rất tích cực`).

### Phương án B: Phân loại 2 nhánh (Multi-Task Learning / Two-Head Model) - Khuyên dùng
* **Cách thức:** Mô hình dự đoán cùng lúc 2 đầu ra cho mỗi câu văn:
  * **Đầu ra 1 (Sentiment Head - 5 classes):** Dự đoán mức độ từ Rất tiêu cực -> Rất tích cực.
  * **Đầu ra 2 (Emotion Head - 6 hoặc 7 classes bao gồm 'None'):** Dự đoán loại cảm xúc cụ thể.
* **Ưu điểm:** Phản ánh đúng bản chất ngôn ngữ tự nhiên (một câu vừa có sắc thái, vừa có trạng thái cảm xúc).

---

## 3. KẾ HOẠCH XÂY DỰNG DỮ LIỆU (5.000 CÂU)

### 3.1. Phân bổ dữ liệu
* **Tổng số lượng:** 5.000 câu.
* **Phân bổ nhãn:**
  * Nếu chọn **11 lớp phẳng**: ~450 câu / nhãn (đảm bảo cân bằng dữ liệu, tránh class imbalance).
  * Phân chia tập dữ liệu:
    * **Train set:** 80% (khoảng 4.000 câu).
    * **Validation set:** 10% (khoảng 500 câu) để theo dõi Early Stopping và chọn Best Checkpoint.
    * **Test set:** 10% (khoảng 500 câu) để đánh giá độc lập.

### 3.2. Cấu trúc file dữ liệu chuẩn (`Data/intent/dataset_5k.json` hoặc `.csv`)
```json
[
  {
    "id": 1,
    "text": "Dịch vụ của quán thực sự tuyệt vời ngoài sức tưởng tượng!",
    "sentiment": "very_positive",
    "emotion": "happiness"
  }
]
```

---

## 4. DANH SÁCH CÔNG VIỆC CẦN LÀM (ACTIONABLE CHECKLIST)

### Giai đoạn 1: Chuẩn bị & Xử lý Dữ liệu
- [ ] **1.1.** Thống nhất ngôn ngữ của tập dữ liệu (Tiếng Anh hay Tiếng Việt).
- [ ] **1.2.** Thống nhất kiến trúc nhãn (Phương án A: 11 lớp phẳng, hay Phương án B: 2 đầu ra Sentiment & Emotion).
- [ ] **1.3.** Thu thập / biên soạn đủ 5.000 câu ngữ cảnh mẫu.
- [ ] **1.4.** Gán nhãn và kiểm tra chất lượng (Data QA) để loại bỏ câu trùng lặp, câu tối nghĩa hoặc gán sai nhãn.
- [ ] **1.5.** Xuất dữ liệu ra file chuẩn (ví dụ: `Data/intent/sentiment_emotion_5k.json`).

### Giai đoạn 2: Cập nhật Cấu hình & Pipeline Dữ liệu
- [ ] **2.1. Cập nhật `DOTA/config.py`**:
  * Định nghĩa lại `LABEL_MAP` với các nhãn mới.
  * Cập nhật `NUM_LABELS` (11 hoặc theo thiết kế mới).
  * Cập nhật đường dẫn file dữ liệu mới (`DATA_JSON = 5k.json`).
  * Điều chỉnh `TRAIN_SPLIT` phù hợp với quy mô 5.000 câu (ví dụ: 4000/5000 = 0.8).
  * Chọn `BERT_MODEL_NAME` phù hợp:
    * Tiếng Anh: `distilbert-base-uncased` hoặc `roberta-base`.
    * Tiếng Việt: `vinai/phobert-base-v2` hoặc `bkai-foundation-models/vietnamese-bi-encoder`.
- [ ] **2.2. Cập nhật `DOTA/data_builder.py`**:
  * Tinh chỉnh hàm đọc và chuẩn hóa dữ liệu đầu vào.
  * Tích hợp kiểm tra cân bằng mẫu giữa các nhãn.
  * Tokenize văn bản với độ dài `MAX_LENGTH` phù hợp.

### Giai đoạn 3: Huấn luyện & Đánh giá Mô hình
- [ ] **3.1. Cập nhật `DOTA/train_bert.py`**:
  * Khởi tạo kiến trúc phân loại mới (`AutoModelForSequenceClassification` hoặc Custom Multi-task Model).
  * Cấu hình hàm mất mát (`Loss function`) và bộ tối ưu (`AdamW`, `Scheduler`).
  * Thực hiện huấn luyện mô hình trên 5.000 câu với cơ chế Early Stopping và Best Model Checkpoint.
- [ ] **3.2. Cập nhật `DOTA/evaluate.py`**:
  * Vẽ Ma trận nhầm lẫn (Confusion Matrix) 11x11 hoặc ma trận cho từng nhóm nhãn.
  * Xuất báo cáo Classification Report (Precision, Recall, F1-Score từng nhãn).
- [ ] **3.3. Cập nhật `DOTA/check_ambiguity.py`**:
  * Kiểm tra các câu có độ tin cậy thấp hoặc dễ nhầm lẫn giữa các mức độ (ví dụ giữa *tích cực* và *rất tích cực*).

### Giai đoạn 4: Tích hợp Hệ thống & Ứng dụng
- [ ] **4.1. Cập nhật `DOTA/predict_bert.py`**:
  * Cập nhật hàm dự đoán trả về nhãn văn bản tiếng Việt/Anh và độ tin cậy (`confidence score`).
- [ ] **4.2. Cập nhật `Code/GUI.py`**:
  * Cập nhật giao diện hiển thị nhãn mới và phản hồi phù hợp tương ứng với từng cảm xúc/sắc thái.
- [ ] **4.3. Kiểm thử tương tác (`interactive_test.py`)**:
  * Thử nghiệm gõ các câu thực tế để kiểm tra độ nhạy và chính xác của mô hình mới.