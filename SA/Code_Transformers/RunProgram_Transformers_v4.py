import os
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import random
import string

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

def but_scan(sentence):
    if 'but' in sentence:
        msg = sentence.split("but", 1)[1]
        return msg
    else:
        return sentence

def reading_sentence(input_file_path, output_file_path, result_file_path,
                     max_sentences=500):
    sentence_count = 0
    matching_sentence_count = 0
    nega1, nega2, posi1, posi2 = 0, 0, 0, 0

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
            # display_negative_percentage(sentence_count, matching_sentence_count, sentence_count)

            if response == f" This is negative 1":
                matching_sentence_count += 1
                nega1 += 1
                # print(f'nega1: {nega1}')
                # print(f'matching_sentence_count: {matching_sentence_count}')
            if response == f" This is negative 2":
                matching_sentence_count += 1
                nega2 += 1
                # print(f'nega2: {nega2}')
                # print(f'matching_sentence_count: {matching_sentence_count}')
            if response == f" This is positive 1":
                matching_sentence_count += 1
                posi1 += 1
                # print(f'posi1: {posi1}')
                # print(f'matching_sentence_count: {matching_sentence_count}')
            if response == f" This is positive 2":
                matching_sentence_count += 1
                posi2 += 1
                # print(f'posi2: {posi2}')
                # print(f'matching_sentence_count: {matching_sentence_count}')

            else:
                continue
                # print("loi k cong duoc")
                error_message = f"Error sentence: {sentence_count}. {target_type} ==> {alternative_type}"
                output_file.write(f"{error_message}\n")
                output_file.write(f"{line}\n")
                output_file.write(f"{predict_class(clean_up_sentence(sentence), intents_data, words, model)}\n")

            # if response == f"1001":
            #     break
            if sentence_count % 5 == 0:
                os.system("cls")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
    print(f'')
    # return exactly result

    return [sentence_count, matching_sentence_count, posi1, posi2, nega1, nega2]

if __name__ == "__main__":
    os.system("cls")

    list1 = reading_sentence(
        "../TestComputer/Computer_posi.txt",
        "../TestComputer/Computer_posi_ERROR.txt",
        "../TestComputer/Result/result_pos_ITEAI.txt"
    )

    list2 = reading_sentence(
        "../TestComputer/Computer_nega.txt",
        "../TestComputer/Computer_nega_ERROR.txt",
        "../TestComputer/Result/result_neg_ITEAI.txt"
    )

    i = list1[0]
    j = list2[0]
    num_p = list1[1]
    num_n = list2[1]
    total = i + j
    accuracy = float((num_n + num_p) / (total))

    print(num_n)
    print(num_p)

    print('The number of positive sentences is: ' + str(num_p) + '\nThe number of negative sentences is: ' + str(num_n))
    print('\nTotal of number : ' + str(total))
    print('\nRate of exactly sentence : ' + str(100 * accuracy) + ' %')
