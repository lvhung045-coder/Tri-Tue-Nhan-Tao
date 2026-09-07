import json


def load_intents(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data['intents']


def classify_intents(sentence, intents):
    words = set(sentence.lower().split())
    results = {}

    for intent in intents:
        matching_words = set()
        for pattern in intent['patterns']:
            if pattern.lower() in words:
                matching_words.add(pattern.lower())

        if matching_words:
            results[intent['tag']] = list(matching_words)

    return results if results else {"No match found": []}


if __name__ == "__main__":
    # Đường dẫn đến file JSON
    file_path = "../Data/intent/transformer_v2.json"

    # Tải dữ liệu từ file JSON
    intents = load_intents(file_path)

    while True:
        # Nhập câu nói từ người dùng
        sentence = input("Nhập câu nói (hoặc 'QUIT' để thoát): ")

        # Kiểm tra xem người dùng có muốn thoát không
        if sentence.upper() == "QUIT":
            print("Thoát chương trình.")
            break

        # Xác định các tag của câu nói
        result = classify_intents(sentence, intents)

        # In ra kết quả
        if result == {"No match found": []}:
            print("Không tìm thấy từ khớp.")
        else:
            for tag, words in result.items():
                print(f"Tag: {tag}, Từ tìm thấy: {', '.join(words)}")
