# Tổng hợp các thay đổi so với Source ban đầu

## 1. Thiết lập môi trường

- **Tạo môi trường ảo** `.venv` tại `c:\NC\nc\.venv`
- **Cài đặt các thư viện**: `nltk`, `keras`, `tensorflow`, `numpy`, `pillow`, `matplotlib`
- **Tải NLTK data**: `punkt`, `punkt_tab`, `wordnet`

---

## 2. Sửa đường dẫn (Tất cả các file)

### Vấn đề gốc
Tất cả các file đều dùng đường dẫn tuyệt đối trỏ đến ổ đĩa `D:` cũ:
```python
# CŨ - không chạy được trên máy khác
intents = json.loads(open(r'D:\DCCNTT13.10.5\bot_tkinter\ChatBot_Ontology\Data\intent\jon.json').read())
```

### Cách sửa
Thêm `BASE_DIR` để tự động tính đường dẫn dựa trên vị trí file:
```python
# MỚI - chạy được ở bất kỳ đâu
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
intents = json.loads(open(os.path.join(BASE_DIR, 'Data', 'intent', 'jon.json')).read())
```

### Danh sách file đã sửa đường dẫn

| # | File | Số đường dẫn đã sửa |
|---|------|---------------------|
| 1 | [RunProgram.py](file:///c:/NC/nc/Code/RunProgram.py) | 7 (data + model + TestComputer) |
| 2 | [RunProgramv2.py](file:///c:/NC/nc/Code/RunProgramv2.py) | 5 (data + model + TestComputer) |
| 3 | [RunProgramv3.py](file:///c:/NC/nc/Code/RunProgramv3.py) | 5 (data + model + TestComputer) |
| 4 | [RunProgramv4.py](file:///c:/NC/nc/Code/RunProgramv4.py) | 5 (data + model + TestComputer) |
| 5 | [GUI.py](file:///c:/NC/nc/Code/GUI.py) | 3 (hình ảnh + icon) |
| 6 | [Training_Bot_v2.py](file:///c:/NC/nc/Code/Training_Bot_v2.py) | 5 (data + pickle + model save) |
| 7 | [writejson.py](file:///c:/NC/nc/Code/writejson.py) | 3 (data + output json) |
| 8 | [weight.py](file:///c:/NC/nc/Code/weight.py) | 1 (json path) |
| 9 | [checklist.py](file:///c:/NC/nc/Code/checklist.py) | 1 (json path) |
| 10 | [Code_Transformers/RunProgram_Transformers_v2.py](file:///c:/NC/nc/Code_Transformers/RunProgram_Transformers_v2.py) | 8 (model + data + TestComputer) |
| 11 | [Code_Transformers/RunProgram_Transformers_v3.py](file:///c:/NC/nc/Code_Transformers/RunProgram_Transformers_v3.py) | 8 (model + data + TestComputer) |
| 12 | [Code_Transformers-20.../RunProgram_Transformers_v2.py](file:///c:/NC/Code_Transformers-20260409T144804Z-3-001/Code_Transformers/RunProgram_Transformers_v2.py) | 8 (bản copy) |
| 13 | [Code_Transformers-20.../RunProgram_Transformers_v3.py](file:///c:/NC/Code_Transformers-20260409T144804Z-3-001/Code_Transformers/RunProgram_Transformers_v3.py) | 8 (bản copy) |

> [!NOTE]
> Còn 6 dòng chứa đường dẫn cũ nhưng đều nằm trong **comment** (`#`) nên không ảnh hưởng đến chương trình.

---

## 3. Sửa lỗi tương thích Keras 3 & NumPy mới (Training_Bot_v2.py)

### 3a. Lỗi `keras.optimizers.legacy` không tồn tại

```diff
 # Import
-from keras.optimizers import SGD
-from keras.optimizers import legacy
+from keras.optimizers import SGD

 # Optimizer
-sgd = legacy.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
+sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
```

> **Lý do:** Keras 3 đã loại bỏ module `legacy`. Tham số `decay` cũng không còn được hỗ trợ.

### 3b. Lỗi NumPy `inhomogeneous array`

```diff
 random.shuffle(training)
-training = np.array(training)
+
+# Tạo mảng numpy tương thích với phiên bản NumPy mới
+dims = (len(training), len(training[0]))
+arr = np.empty(dims, dtype=object)
+for i in range(len(training)):
+    arr[i] = training[i]
+training = np.array(arr)
```

> **Lý do:** NumPy 2.x không cho phép tạo mảng từ danh sách có kích thước không đồng nhất (`bag` và `output_row` có độ dài khác nhau).

### 3c. Thiếu `import os`

```diff
 import nltk
+import os
 from nltk.stem import WordNetLemmatizer
```

> **Lý do:** Cần `os` để sử dụng `os.path.join()` và `os.path.dirname()`.

---

## 4. File không thay đổi

Các file sau **không cần sửa** vì không chứa đường dẫn cố định hoặc là file dữ liệu:
- `Data/intent/jon.json` (dữ liệu)
- `Data/*.pkl` (dữ liệu pickle)
- `Code/Model/*.h5` (model đã train)
- `Code/trongso.py`
- `Training_Bot.py` (phiên bản cũ, không sử dụng)

---

## 5. Cách chạy chương trình

```powershell
# Di chuyển đến thư mục dự án
cd c:\NC\nc

# Chạy giao diện Chatbot
.venv\Scripts\python.exe Code\GUI.py

# Huấn luyện lại mô hình
.venv\Scripts\python.exe Code\Training_Bot_v2.py
```
