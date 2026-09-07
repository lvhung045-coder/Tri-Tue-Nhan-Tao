import json
import random
import string

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Đường dẫn tới mô hình và tokenizer đã lưu
model_path = './Model/BertModel'
data_path = '../Data/intent/transformer_v2.json'

# Đọc dữ liệu từ tệp JSON
data = json.loads(open(data_path).read())

# Tải mô hình và tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Chuẩn bị label map
label_map = {intent['tag']: i for i, intent in enumerate(data['intents'])}
reverse_label_map = {v: k for k, v in label_map.items()}

def clean_up_sentence(sentence):
    return "".join(filter(lambda x: x in string.ascii_letters + ' ', sentence))

def predict_class(sentence, model, tokenizer):
    cleaned_sentence = clean_up_sentence(sentence)
    inputs = tokenizer(cleaned_sentence, return_tensors='pt', padding=True, truncation=True, max_length=128)
    outputs = model(**inputs)
    _, prediction = torch.max(outputs.logits, dim=1)
    return reverse_label_map[prediction.item()]

def get_response(ints, intents_data):
    tag = ints
    list_of_intents = intents_data['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result

def chatbot_response(msg):
    intent = predict_class(msg, model, tokenizer)
    response = get_response(intent, data)
    return response

# Ví dụ sử dụng
if __name__ == "__main__":
    while True:
        msg = input("You: ")
        if msg.lower() == "quit":
            break
        res = chatbot_response(msg)
        print(f"Bot: {res}")
