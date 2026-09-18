# Báo Cáo So Sánh: Trạng Thái Hệ Thống Lúc Đầu vs Hiện Tại

Tài liệu này tóm tắt sự khác biệt về mặt kiến trúc, dữ liệu và hiệu năng giữa phiên bản Chatbot đầu tiên của dự án và phiên bản trí tuệ nhân tạo (AI) hiện tại.

---

## 1. Kiến Trúc AI (Core Architecture)

| Tiêu chí | Lúc Đầu (Phiên bản Keras) | Hiện Tại (Phiên bản BERT Multi-Task) |
| :--- | :--- | :--- |
| **Thuật toán chính** | Mạng nơ-ron truyền thống (Sequential Neural Network) sử dụng Keras/TensorFlow. | Kiến trúc Transformers tiên tiến (BERT/PhoBERT) sử dụng HuggingFace PyTorch. |
| **Cách hiểu ngôn ngữ** | **Bag of Words (BoW):** Chỉ đếm tần suất xuất hiện của từ ngữ, đánh mất hoàn toàn ngữ cảnh và thứ tự từ trong câu. | **Contextual Embeddings (Attention):** Hiểu được ngữ cảnh, thứ tự từ và ý nghĩa thực sự của câu (VD: hiểu được từ lóng, câu phức). |
| **Mục tiêu học (Task)** | Single-Task: Chỉ dự đoán ra 1 nhãn duy nhất (Intents). | **Multi-Task Learning:** Mô hình có "2 cái đầu" (2 Linear Heads), vừa học phân loại **Sắc thái** (5 mức độ), vừa học phân loại **Cảm xúc** (6 loại) cùng một lúc. |

---

## 2. Quản Lý Dữ Liệu & Tiền Xử Lý (Data & Preprocessing)

| Tiêu chí | Lúc Đầu | Hiện Tại |
| :--- | :--- | :--- |
| **Định dạng dữ liệu** | Phân mảnh: Chứa trong nhiều file `.pkl`, `.csv`, `.txt` và file JSON cũ với cấu trúc mảng lồng nhau phức tạp. | Tập trung: Tất cả dữ liệu học nằm gọn trong 1 file `dataset_5k_raw.json` duy nhất với cấu trúc bảng (Flat JSON) rất dễ đọc. |
| **Kỹ thuật tách từ (Tokenizer)** | Dùng thư viện `NLTK`, phải thủ công cắt chữ, chuẩn hóa từ vựng (Lemmatization) và tự lưu từ điển. | Dùng `AutoTokenizer` của BERT: Tự động dùng thuật toán BPE/WordPiece cắt từ siêu tốc, kho từ vựng hàng chục ngàn từ đã được nhúng sẵn, không cần file từ điển ngoài. |

---

## 3. Cấu Trúc Mã Nguồn (Codebase Structure)

| Tiêu chí | Lúc Đầu | Hiện Tại |
| :--- | :--- | :--- |
| **Sự sắp xếp** | Hơn 15 file code rải rác lộn xộn ở thư mục `Code/` (như `RunProgramv2`, `v3`, `v4`, `Training_bot`, `weight.py`...). | Cực kỳ gọn gàng, chia module rõ ràng: <br> - `Data/`: Chứa dữ liệu <br> - `DOTA/`: Chứa não bộ AI (`train_bert.py`, `predict_bert.py`, `config.py`) <br> - `Code/`: Chỉ chứa giao diện `GUI.py`. |
| **Giao diện (GUI)** | Thiếu cơ chế kiểm duyệt, nhập số hay ký tự đặc biệt AI vẫn cố gắng dự đoán bừa. | Đã bổ sung bộ lọc thông minh: Chặn đứng các câu vô nghĩa, toàn số hoặc toàn ký tự đặc biệt để bảo vệ AI. Giao diện hiển thị phản hồi có ngắt dòng và màu sắc dễ nhìn. |

---

## 4. Tổng Kết Giá Trị Nâng Cấp

Quá trình tái cấu trúc đã mang lại những lợi ích to lớn cho dự án:
1. **Hiệu năng nhận thức vượt trội:** Mô hình chuyển từ việc "đoán mò từ khóa" sang việc "đọc hiểu ngữ cảnh", đồng thời giải quyết được bài toán phân tích 2 khía cạnh độc lập (Sắc thái & Cảm xúc) chỉ trong 1 lần dự đoán.
2. **Khả năng mở rộng (Scalability):** Dễ dàng chuyển đổi sang học **Tiếng Việt** chỉ bằng cách sửa đúng 1 dòng code tên mô hình (`vinai/phobert-base-v2` hoặc `xlm-roberta-base`) mà không cần đập đi xây lại.
3. **Tính chuyên nghiệp (Clean Code):** Hệ thống không còn file rác. Việc đưa file dự án đi báo cáo đồ án, chạy trên máy chủ khác hoặc tải lên Github trở nên cực kỳ nhẹ nhàng.
