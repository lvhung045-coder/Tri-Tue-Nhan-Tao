# -*- coding: utf-8 -*-
import os
import sys
import time

# Thêm đường dẫn để import
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(BASE_DIR, 'Code'))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'DOTA'))

from RunProgram import chatbot_response as bow_response
from DOTA.predict_bert import BertPredictor

def compare():
    print("=" * 80)
    print(f"{'CÂU TEST (NGỮ CẢNH PHỨC TẠP)':<40} | {'MÔ HÌNH CŨ (BoW)':<15} | {'MÔ HÌNH MỚI (BERT)':<15}")
    print("-" * 80)

    # Danh sách các câu test "bẫy" mô hình cũ
    test_cases = [
        "I don't like this product",
        "This is not bad at all",
        "I am not happy with the service",
        "It was never good",
        "I wouldn't say it's positive",
        "This is absolutely awful",
        "Amazing quality for the price",
        "Stop this offensive behavior",
    ]

    predictor = BertPredictor()
    
    # Map phản hồi về tag để dễ so sánh
    # BoW trả về "This is negative_1", BERT trả về tag "negative_1"
    
    for sentence in test_cases:
        # 1. BoW Result
        start_bow = time.time()
        res_bow = bow_response(sentence)
        time_bow = (time.time() - start_bow) * 1000

        # 2. BERT Result
        start_bert = time.time()
        tag_bert, conf_bert = predictor.predict(sentence)
        time_bert = (time.time() - start_bert) * 1000

        # Rút gọn kết quả hiển thị
        display_bow = res_bow.replace("This is ", "")
        display_bert = tag_bert

        # Đánh dấu đúng sai (Giả định câu 1,3,4,5,6,8 là negative, 2,7 là positive)
        print(f"{sentence:<40} | {display_bow:<15} | {display_bert:<15}")

    print("-" * 80)
    print("\nPHÂN TÍCH ƯU/NHƯỢC ĐIỂM:")
    print("1. Tốc độ: BoW chạy nhanh hơn vì chỉ tính toán xác suất từ vựng đơn giản.")
    print("2. Thông minh: BERT hiểu được sự liên kết giữa các từ.")
    print("   - Ví dụ: 'don't like' -> BERT hiểu là phủ định của 'like' nên đoán đúng Negative.")
    print("   - BoW: Thường bị nhầm nếu từ 'like' có trọng số cao hơn từ 'don't'.")
    print("=" * 80)

if __name__ == "__main__":
    compare()
