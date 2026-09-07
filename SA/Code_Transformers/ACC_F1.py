# from sklearn.metrics import accuracy_score
#
# def calculate_accuracy(file_path):
#     TP = TN = FP = FN = 0
#
#     with open(file_path, 'r', encoding='utf-8') as file:
#         lines = file.readlines()
#         for i in range(len(lines) - 1):
#             if lines[i].strip() in ['po1', 'po2', 'ne1', 'ne2']:
#                 predicted_label = lines[i].strip()
#                 true_label = lines[i + 1].strip()
#
#                 if predicted_label == true_label:
#                     if predicted_label in ['po1', 'po2']:
#                         TP += 1  # True Positive
#                     else:
#                         TN += 1  # True Negative
#                 else:
#                     if predicted_label in ['po1', 'po2']:
#                         FP += 1  # False Positive
#                     else:
#                         FN += 1  # False Negative
#
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     print(f"Accuracy: {accuracy}")
#
# # Ví dụ sử dụng
# file_path = '../TestComputer/Real_EN/test.txt'
# calculate_accuracy(file_path)


# from sklearn.metrics import accuracy_score, f1_score
#
# # Đọc dữ liệu từ file
# with open('../TestComputer/Real_EN/test.txt', 'r') as file:
#     data = file.read()
#
# # Tách các đoạn dữ liệu
# segments = data.split('==')
#
# # Lưu trữ giá trị po1, po2, ne1, ne2
# y_true = []
# y_pred = []
#
# # print(y_true)
# # print(y_pred)
# # print(segments)
#
# # Phân tích từng đoạn để lấy giá trị
# for segment in segments:
#     lines = segment.strip().split('\n')
#     if len(lines) >= 4:
#         true_label = lines[-2].strip()  # Lấy nhãn đúng
#         pred_label = lines[-1].strip()  # Lấy nhãn dự đoán
#         y_true.append(true_label)
#         y_pred.append(pred_label)
#         print(true_label)
#         print(pred_label)
#
# # Chuyển đổi nhãn về dạng nhị phân
# y_true_bin = [1 if label.startswith('po2') else 0 for label in y_true]
# y_pred_bin = [1 if label.startswith('po1') else 0 for label in y_pred]
# x_true_bin = [1 if label.startswith('ne2') else 0 for label in y_true]
# x_pred_bin = [1 if label.startswith('ne1') else 0 for label in y_pred]
#
# print("\n=========================")
# print(y_true_bin)
# print(y_pred_bin)
# print(x_true_bin)
# print(x_pred_bin)
#
# # Tính accuracy và F1-score
# accuracy = accuracy_score(y_true_bin, y_pred_bin)
# f1 = f1_score(y_true_bin, y_pred_bin)
#
# print(f'Accuracy: {accuracy}')
# print(f'F1-score: {f1}')

# def extract_last_two_lines(file_path):
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     # Lưu trữ các cặp dòng cuối cùng từ mỗi vùng dữ liệu
#     last_two_lines = []
#     current_region = []
#
#     for line in lines:
#         line = line.strip()
#
#         if line.startswith("=="):
#             # Nếu bắt đầu một vùng mới, lưu 2 dòng cuối cùng của vùng trước (nếu có)
#             if len(current_region) >= 2:
#                 last_two_lines.append(current_region[-2:])
#             current_region = []  # Khởi tạo lại vùng dữ liệu mới
#
#         if line:  # Bỏ qua dòng trống
#             current_region.append(line)
#
#     # Lưu 2 dòng cuối cùng của vùng dữ liệu cuối cùng (nếu có)
#     if len(current_region) >= 2:
#         last_two_lines.append(current_region[-2:])
#
#     return last_two_lines
#
#
# # Đường dẫn đến file txt
# file_path = '../TestComputer/Real_EN/test.txt'
# last_two_lines_per_region = extract_last_two_lines(file_path)
#
# # In kết quả
# for index, lines in enumerate(last_two_lines_per_region):
#     print(f"Vùng {index + 1}:")
#     for line in lines:
#         print(line)
#     print()

# ============================================================================================
# def calculate_metrics(file_path):
#     TP = TN = FP = FN = 0  # Khởi tạo các biến đếm
#
#     with open(file_path, 'r') as file:
#         lines = file.readlines()
#
#     current_region = []
#
#     for line in lines:
#         line = line.strip()
#
#         if line.startswith("=="):
#             # Nếu bắt đầu một vùng mới, kiểm tra các dòng trong vùng trước (nếu có)
#             if len(current_region) >= 2:
#                 label_1 = current_region[-2]
#                 label_2 = current_region[-1]
#
#                 # Kiểm tra các điều kiện cho TP, TN, FP, FN
#                 if label_1 == "po1" and label_2 == "po1" or label_1 == "po2" and label_2 == "po2":
#                     TP += 1
#                 elif label_1 == "ne1" and label_2 == "ne1" or label_1 == "ne2" and label_2 == "ne2":
#                     TN += 1
#                 elif label_1 != label_2 and (label_2 == "po1" or label_2 == "po2"):
#                     FP += 1
#                 elif label_1 != label_2 and (label_2 == "ne1" or label_2 == "ne2"):
#                     FN += 1
#
#             current_region = []  # Khởi tạo lại vùng dữ liệu mới
#
#         if line:  # Bỏ qua dòng trống
#             current_region.append(line)
#
#     # Kiểm tra vùng dữ liệu cuối cùng (nếu có)
#     if len(current_region) >= 2:
#         label_1 = current_region[-2]
#         label_2 = current_region[-1]
#
#         # Kiểm tra các điều kiện cho TP, TN, FP, FN
#         if label_1 == "po1" and label_2 == "po1" or label_1 == "po2" and label_2 == "po2":
#             TP += 1
#         elif label_1 == "ne1" and label_2 == "ne1" or label_1 == "ne2" and label_2 == "ne2":
#             TN += 1
#         elif label_1 != label_2 and (label_2 == "po1" or label_2 == "po2"):
#             FP += 1
#         elif label_1 != label_2 and (label_2 == "ne1" or label_2 == "ne2"):
#             FN += 1
#
#     return TP, TN, FP, FN
#
#
# # Đường dẫn đến file txt
# file_path = '../TestComputer/Real_EN/test.txt'
# TP, TN, FP, FN = calculate_metrics(file_path)
#
# # In kết quả
# print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")
#


def calculate_metrics(file_path):
    TP = TN = FP = FN = 0  # Khởi tạo các biến đếm

    with open(file_path, 'r') as file:
        lines = file.readlines()

    current_region = []

    for line in lines:
        line = line.strip()

        if line.startswith("=="):
            # Nếu bắt đầu một vùng mới, kiểm tra các dòng trong vùng trước (nếu có)
            if len(current_region) >= 2:
                label_1 = current_region[-2]
                label_2 = current_region[-1]

                # Kiểm tra các điều kiện cho TP, TN, FP, FN
                if (label_1 == "po1" and label_2 == "po1") or (label_1 == "po2" and label_2 == "po2"):
                    TP += 1
                elif (label_1 == "ne1" and label_2 == "ne1") or (label_1 == "ne2" and label_2 == "ne2"):
                    TN += 1
                elif label_1 != label_2 and (label_2 == "po1" or label_2 == "po2"):
                    FP += 1
                elif label_1 != label_2 and (label_2 == "ne1" or label_2 == "ne2"):
                    FN += 1

            current_region = []  # Khởi tạo lại vùng dữ liệu mới

        if line:  # Bỏ qua dòng trống
            current_region.append(line)

    # Kiểm tra vùng dữ liệu cuối cùng (nếu có)
    if len(current_region) >= 2:
        label_1 = current_region[-2]
        label_2 = current_region[-1]

        # Kiểm tra các điều kiện cho TP, TN, FP, FN
        if (label_1 == "po1" and label_2 == "po1") or (label_1 == "po2" and label_2 == "po2"):
            TP += 1
        elif (label_1 == "ne1" and label_2 == "ne1") or (label_1 == "ne2" and label_2 == "ne2"):
            TN += 1
        elif label_1 != label_2 and (label_2 == "po1" or label_2 == "po2"):
            FP += 1
        elif label_1 != label_2 and (label_2 == "ne1" or label_2 == "ne2"):
            FN += 1

    return TP, TN, FP, FN


def calculate_accuracy_f1(TP, TN, FP, FN):
    # Tính toán Accuracy
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0

    # Tính toán Precision và Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    # Tính toán F1-score
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    print(f'accuracy: {accuracy}')
    print(f'precision: {precision}')
    print(f'recall" {recall}')
    print(f'f1_score: {f1_score}')

    return accuracy, f1_score


# Đường dẫn đến file txt
file_path = '../TestComputer/Real_EN/ReadAcc.txt'
TP, TN, FP, FN = calculate_metrics(file_path)
print(f"TP: {TP}, TN: {TN}, FP: {FP}, FN: {FN}")

# Tính toán Accuracy và F1
accuracy, f1_score = calculate_accuracy_f1(TP, TN, FP, FN)

# In kết quả
print(f"Accuracy: {accuracy*100:.3f}, F1-score: {f1_score*100:.3f}")
