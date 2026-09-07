import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random
import string

# Base directory = thư mục nc/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đường dẫn tới mô hình và tokenizer đã lưu
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Model', 'BertModel')
data_path = os.path.join(BASE_DIR, 'Data', 'intent', 'transformer_v2.json')

# Đọc dữ liệu từ tệp JSON
data = json.loads(open(data_path).read())

# Tải mô hình và tokenizer
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# Chuẩn bị label map
label_map = {intent['tag']: i for i, intent in enumerate(data['intents'])}
print(label_map)
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
    print(tag)
    list_of_intents = intents_data['intents']
    # print(f' list {list_of_intents}')
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            print(f'result {result}')
            break
    return result

def chatbot_response(msg):
    intent = predict_class(msg, model, tokenizer)
    response = get_response(intent, data)
    return response

def but_scan(sentence):
    if 'but' in sentence:
        msg = sentence.split("but", 1)[1]
        return msg
    else:
        return sentence

def reading_sentence(input_file_path, output_file_path, result_file_path, target_types,
                     max_sentences=500):
    sentence_count = 0
    matching_sentence_count = 0
    lv_counts = {target_type: 0 for target_type in target_types}

    with open(input_file_path, "r") as input_file, open(output_file_path, "w") as output_file, open(result_file_path, "w") as result_file:
        for line in input_file:
            sentence = line.strip().split("##", 1)[1]
            sentence_count += 1

            if sentence_count > max_sentences:
                break

            print("|| ============================== " + str(sentence_count) + " ============================== || ")
            print("You: " + sentence + "\n")
            response = chatbot_response(but_scan(sentence))
            print("Bot: " + response + "")

            for target_type in target_types:
                if response == f"This is {target_type}":
                    matching_sentence_count += 1
                    lv_counts[target_type] += 1
                    break

            if response == f"1001":
                break
            if sentence_count % 5 == 0:
                os.system("cls")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")

    print(matching_sentence_count)
    return [sentence_count, matching_sentence_count] + [lv_counts[target_type] for target_type in target_types]

# Ví dụ sử dụng
if __name__ == "__main__":
    os.system("cls")

    target_types_positive = ["positive"]
    list1 = reading_sentence(
        os.path.join(BASE_DIR, "TestComputer", "Computer_posi.txt"),
        os.path.join(BASE_DIR, "TestComputer", "Computer_posi_ERROR.txt"),
        os.path.join(BASE_DIR, "TestComputer", "Result", "result_pos_ITEAI.txt"),
        target_types_positive
    )

    target_types_negative = ["negative_1", "negative_2", "negative_3"]
    list2 = reading_sentence(
        os.path.join(BASE_DIR, "TestComputer", "Computer_nega.txt"),
        os.path.join(BASE_DIR, "TestComputer", "Computer_nega_ERROR.txt"),
        os.path.join(BASE_DIR, "TestComputer", "Result", "result_neg_ITEAI.txt"),
        target_types_negative
    )

    i = list1[0]
    j = list2[0]
    num_p = list1[1]
    num_n = list2[1]
    total = i + j - 2
    accuracy = float((num_n + num_p) / (total))
    print('The number of positive sentences is: ' + str(num_p) + '\nThe number of negative sentences is: ' + str(num_n))
    print('\nTotal of number : ' + str(total))
    print('\nRate of exactly sentence : ' + str(100 * accuracy) + ' %')
