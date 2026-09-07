import json
import os

from nltk.stem import WordNetLemmatizer
import string
import numpy as np
import random
import nltk

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tuyệt đối của file JSON
json_file_path = os.path.join(BASE_DIR, "Data", "intent", "jon.json")

# Đọc dữ liệu từ file JSON
with open(json_file_path, 'r') as file:
    intents_data = json.load(file)

# Xử lý dữ liệu
words = []
classes = []
patterns = []

for intent in intents_data['intents']:
    tag = intent['tag']
    classes.append(tag)

    for pattern in intent['patterns']:
        # Tách từ và gán trọng số
        for word in pattern.split():
            weight = intent.get('weight', 1)  # Mặc định trọng số là 1 nếu không có trường 'weight'
            words.append((lemmatizer.lemmatize(word.lower()), weight))
            patterns.append((lemmatizer.lemmatize(word.lower()), tag))

# Loại bỏ các từ trùng lặp
words = list(set(words))
patterns = list(set(patterns))

# In ra một số ví dụ để kiểm tra
print("Words with weights:")
print(words[:5])
print("\nPatterns with tags:")
print(patterns[:5])
