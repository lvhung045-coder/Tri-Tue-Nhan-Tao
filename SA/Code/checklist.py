import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tới tệp JSON
file_path = os.path.join(BASE_DIR, "Data", "intent", "jon.json")

# Hàm để tìm các từ giống nhau trong các patterns của các intent
def find_duplicate_words(file_path):
    # Mở tệp JSON và đọc dữ liệu
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    # Danh sách chứa tất cả các từ
    all_words = []

    # Lặp qua từng intent
    for intent in data['intents']:
        # Lặp qua từng pattern trong intent và thêm từ vào danh sách nếu chưa có trong đó
        for pattern in intent['patterns']:
            words = pattern.split()
            for word in words:
                if word not in all_words:
                    all_words.append(word)
                else:
                    print(f"Từ '{word}' được tìm thấy trong patterns của nhiều intent.")

# Gọi hàm để tìm các từ giống nhau
find_duplicate_words(file_path)
