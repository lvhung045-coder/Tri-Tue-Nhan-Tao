# # import json
# # import numpy as np
# # from keras.models import load_model
# # import pickle
# #
# #
# # def load_and_predict_model(model_path, words, negative_words, threshold):
# #     # Load mô hình từ file h5
# #     model = load_model(model_path)
# #
# #     # Tạo một list để lưu các từ tiêu cực và điểm số tiêu cực
# #     negative_words_scores = []
# #
# #     # Duyệt qua các từ
# #     for word in words:
# #         # Chuyển từ thành vectơ bag of words
# #         bag = [1 if w == word else 0 for w in words]
# #
# #         # Dự đoán điểm số tiêu cực của từ
# #         negative_score = model.predict(np.array([bag]))[0][0]
# #
# #         # Nếu điểm số tiêu cực vượt ngưỡng, thêm từ vào danh sách negative_words_scores
# #         if negative_score > threshold:
# #             negative_words_scores.append({"word": word, "negative_score": negative_score})
# #
# #     # Trả về danh sách các từ tiêu cực và điểm số tiêu cực
# #     return negative_words_scores
# #
# #
# # def save_to_json(negative_words_scores, target_path, weight=2):
# #     # Tạo cấu trúc dữ liệu JSON
# #     negative_words_2 = {
# #         "intents": [
# #             {
# #                 "tag": "negative",
# #                 "patterns": [],  # Danh sách các từ tiêu cực được tìm thấy
# #                 "responses": [" This is negative"],  # Câu trả lời
# #                 "context": [],
# #                 "weight": weight  # Gắn trọng số
# #             }
# #         ]
# #     }
# #
# #     # Lọc các từ tiêu cực và ghi vào danh sách patterns
# #     for word_score in negative_words_scores:
# #         negative_words_2["intents"][0]["patterns"].append(word_score["word"])
# #
# #     # Ghi nội dung vào tệp đích
# #     with open(target_path, 'w') as target_file:
# #         json.dump(negative_words_2, target_file, indent=4)
# #
# #
# # # Đường dẫn tới mô hình đã huấn luyện và các tệp dữ liệu
# # model_path = '../Model/OWLChatModel_v2.h5'
# # words_path = '../Data/words.pkl'
# # target_path = '../Data/intent/02.json'
# #
# # # Đọc danh sách từ từ tệp words.pkl
# # words = pickle.load(open(words_path, 'rb'))
# #
# # # Định nghĩa ngưỡng để lọc các từ tiêu cực
# # threshold = 0.2
# #
# # # Tìm và dự đoán điểm số tiêu cực của các từ
# # negative_words_scores = load_and_predict_model(model_path, words, threshold)
# #
# # # Lưu các từ tiêu cực vào file 02.json với trọng số 2
# # save_to_json(negative_words_scores, target_path)
#
#
# # trọng số 2
#
# import json
# import random
#
#
# def cut_and_save_to_json(source_path, target_path, weight=2):
#     # Đọc nội dung từ file nguồn
#     with open(source_path, 'r') as source_file:
#         source_data = json.load(source_file)
#
#     # Lấy danh sách các từ tiêu cực từ file nguồn
#     negative_patterns = source_data["intents"][0]["patterns"]
#
#     # Lấy ngẫu nhiên một nửa số từ
#     selected_patterns = random.sample(negative_patterns, len(negative_patterns) // 2)
#
#     # Loại bỏ các từ được chọn khỏi danh sách từ tiêu cực ban đầu
#     remaining_patterns = [pattern for pattern in negative_patterns if pattern not in selected_patterns]
#
#     # Tạo cấu trúc dữ liệu JSON cho file nguồn (file 01.json) với danh sách từ tiêu cực đã được cắt
#     source_data["intents"][0]["patterns"] = remaining_patterns
#
#     # Ghi nội dung mới vào file nguồn
#     with open(source_path, 'w') as source_file:
#         json.dump(source_data, source_file, indent=4)
#
#     # Tạo cấu trúc dữ liệu JSON cho file đích (file 02.json) với danh sách từ tiêu cực đã được cắt
#     target_data = {
#         "intents": [
#             {
#                 "tag": "negative",
#                 "patterns": selected_patterns,
#                 "responses": [" This is negative"],
#                 "context": [],
#                 "weight": weight
#             }
#         ]
#     }
#
#     # Ghi nội dung vào file đích
#     with open(target_path, 'w') as target_file:
#         json.dump(target_data, target_file, indent=4)
#
#
# # Đường dẫn tới file nguồn và file đích
# source_path = '../Data/intent/03.json'
# target_path = '../Data/intent/003.json'
#
# # Gọi hàm để cắt và lưu vào file 02.json
# cut_and_save_to_json(source_path, target_path)
#
#
#
