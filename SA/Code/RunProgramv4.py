import os
import random
import string
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import json

lemmatizer = WordNetLemmatizer()

# Base directory = thư mục cha của Code/ (tức thư mục nc/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

intents_data = json.loads(open(os.path.join(BASE_DIR, 'Data', 'intent', 'jon.json')).read())
model = load_model(os.path.join(BASE_DIR, 'Code', 'Model', 'OWLChatModel_v4.h5'))

def clean_up_sentence(sentence):
    sentence = "".join(filter(lambda x: x in string.ascii_letters + ' ', sentence))
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except:
        print("Invalid sentence!")
        return []

    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

is_p = True
def bow(sentence, intents_data, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for intent in intents_data['intents']:
        for pattern in intent['patterns']:
            if pattern in sentence_words:
                index = words.index(pattern)
                if 'weight' in intent:
                    weight = intent['weight']
                    bag[index] += weight
        if is_p:
            index = words.index(pattern)
            weight = 1
            bag[index] += weight
    return np.array(bag)




def predict_class(sentence, intents_data, words, model):
    p = bow(sentence, intents_data, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": intents_data['intents'][r[0]]['tag'], "probability": str(r[1])})
    return return_list


def get_response(ints, intents_data):
    tag = ints[0]['intent']
    list_of_intents = intents_data['intents']
    for intent in list_of_intents:
        if intent['tag'] == tag:
            result = random.choice(intent['responses'])
            break
    return result


def chatbot_response(msg):
    words = [pattern for intent in intents_data['intents'] for pattern in intent['patterns']]
    ints = predict_class(msg, intents_data, words, model)
    res = get_response(ints, intents_data)
    return res


def but_scan(sentence):
    if 'but' in sentence:
        msg = sentence.split("but", 1)[1]
        return msg
    else:
        return sentence


def reading_sentence(input_file_path, output_file_path, result_file_path, target_type_1, target_type_2, target_type_3,
                     max_sentences=500):
    sentence_count = 0
    matching_sentence_count = 0
    lv1, lv2, lv3 = 0, 0, 0

    with open(os.path.join(BASE_DIR, "TestComputer", input_file_path),
              "r") as input_file, open(
            os.path.join(BASE_DIR, "TestComputer", "Result", output_file_path),
            "w") as output_file, open(
        os.path.join(BASE_DIR, "TestComputer", "Result", result_file_path), "w") as result_file:
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

            if response == f"This is {target_type_1}":
                matching_sentence_count += 1
                lv1 += 1
                if response == f"This is {target_type_2}":
                    matching_sentence_count += 1
                    lv2 += 1
                    if response == f"This is {target_type_3}":
                        matching_sentence_count += 1
                        lv3 += 1

            else:
                continue
                # error_message = f"Error sentence: {sentence_count}. {target_type} ==> {alternative_type}"
                # output_file.write(f"{error_message}\n")
                # output_file.write(f"{line}\n")
                # output_file.write(f"{predict_class(clean_up_sentence(sentence), intents_data, words, model)}\n")

            if response == f"1001":
                break
            if sentence_count % 5 == 0:
                os.system("cls")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")

    # return exactly result
            print(matching_sentence_count)

    return [sentence_count, matching_sentence_count, lv1, lv2, lv3]


if __name__ == "__main__":
    os.system("cls")
    words = [pattern for intent in intents_data['intents'] for pattern in intent['patterns']]
    list1 = reading_sentence("Computer_posi.txt", "Computer_posi_ERROR.txt", "result_pos_ITEAI.txt", "positive_1", "positive_2", "positive_3"
                             )
    list2 = reading_sentence("Computer_nega.txt", "Computer_nega_ERROR.txt", "result_neg_ITEAI.txt", "negative_1", "negative_2", "negative_3"
                             )
    sent = "## The problems came from my video card ."
    bow(sent, intents_data, words,True)
    print(list1, list2,"list")
    i = list1[0]
    j = list2[0]
    num_p = list1[1]
    num_n = list2[1]
    total = i + j - 2
    accuracy = float((num_n + num_p) / (total))
    print('The number of positive sentences is: ' + str(num_p) + '\nThe number of negative sentences is: ' + str(num_n))
    print('\nTotal of number : ' + str(total))
    print('\nRate of exactly sentence : ' + str(100 * accuracy) + ' %')
    #delete 342/vector

    # In ra số lượng câu
    # total_positive_sentences = list1[0]
    # total_negative_sentences = list2[0]
    # total_number = total_positive_sentences + total_negative_sentences
    # print("The number of positive sentences is:", total_positive_sentences)
    # print("The number of negative sentences is:", total_negative_sentences)
    # print("Total of number is:", total_number)

    # # Tính tỷ lệ chính xác và in ra
    # print(list1[1])
    # matching_positive_sentences = list1[1]
    # matching_negative_sentences = list2[1]
    # matching_total = matching_positive_sentences + matching_negative_sentences
    # rate_of_exactly_sentence = matching_total / total_number * 100
    # print("Rate of exactly sentence is: {:.2f}%".format(rate_of_exactly_sentence))

    # bow