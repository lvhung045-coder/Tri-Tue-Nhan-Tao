import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
import pickle
import string
import random
import numpy as np
import time
import os
import json
import random
import matplotlib.pyplot as plt
import nltk
# nltk.download('wordnet')
tic = time.perf_counter()  # Start Time

from keras.models import load_model

# Base directory = thư mục cha của Code/ (tức thư mục nc/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

model = load_model(os.path.join(BASE_DIR, 'Code', 'Model', 'OWLChatModel_v4.h5'))
# model = load_model('Model/chatbot_model.h5')
# model = load_model('Model/ACCLE_model.h5')
# model = load_model('Model/ITEAI_model.h5')


intents = json.loads(open(os.path.join(BASE_DIR, 'Data', 'intent', 'jon.json')).read())
words = pickle.load(open(os.path.join(BASE_DIR, 'Data', 'words.pkl'), 'rb'))
classes = pickle.load(open(os.path.join(BASE_DIR, 'Data', 'classes.pkl'), 'rb'))
negation_words = pickle.load(open(os.path.join(BASE_DIR, 'Data', 'negation_words.pkl'), 'rb'))

is_negated = False

# def clean_up_sentence(sentence):
#     # tokenize the pattern - split words into array
#     print(sentence)
#     sentence_words = nltk.word_tokenize(sentence)
#     # stem each word - create short form for word
#     sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
#     return sentence_words



def clean_up_sentence(sentence):
    global is_negated
    is_negated = False  # Reset mỗi lần gọi

    # Kiểm tra từ phủ định TRƯỚC KHI tokenize (vì nltk tách don't → do + n't)
    sentence_lower = sentence.lower()
    for neg_word in negation_words:
        if neg_word in sentence_lower:
            is_negated = True
            break

    # Filter special characters
    sentence = "".join(filter(lambda x: x in string.ascii_letters + ' ', sentence))

    # Word Separation
    try:
        sentence_words = nltk.word_tokenize(sentence)
    except:
        print("Invalid sentence!")
        return []

    # Stem words
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    if is_negated:
        bag = [-1*w for w in bag]
    return (np.array(bag))

# def bow(sentence, json_paths, show_details=True):
#     # Initialize an empty dictionary to store words and their weights
#     word_weights = {}
#
#     # Iterate through each JSON file path
#     for json_path in json_paths:
#         # Read the data from JSON file
#         with open(json_path, 'r') as f:
#             intents = json.load(f)
#
#         # Iterate through each intent in the JSON data
#         for intent in intents['intents']:
#             # Extract words and weight from the intent
#             words = intent['patterns']
#             weight = intent['weight']
#
#             # Update the word_weights dictionary with words and their weights
#             for word in words:
#                 word_weights[word] = weight
#
#     # Initialize an empty dictionary to store bag of words
#     bag = {}
#
#     # Tokenize the sentence
#     sentence_words = sentence.split()
#
#     # Iterate through each word in the sentence
#     for word in sentence_words:
#         # Check if the word exists in the word_weights dictionary
#         if word in word_weights:
#             # If yes, assign the weight of the word to the bag of words
#             bag[word] = word_weights[word]
#
#     if show_details:
#         print("Bag of words:", bag)
#
#     return bag
#
#
# # Example usage:
# json_paths1 = r"D:\DCCNTT13.10.5\bot_tkinter\ChatBot_Ontology\Data\intent\01.json"
# json_paths2 = r"D:\DCCNTT13.10.5\bot_tkinter\ChatBot_Ontology\Data\intent\02.json"
# json_paths3 = r"D:\DCCNTT13.10.5\bot_tkinter\ChatBot_Ontology\Data\intent\03.json"
#
# result = bow("sentence", [json_paths1, json_paths2, json_paths3])



def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    result = "Xin lỗi, tôi không hiểu câu hỏi của bạn."
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


def but_scan(sentence):
    if 'but' in sentence:
        msg = sentence.split("but", 1)[1]
        return msg
    else:
        return sentence

def shuffle_lines(file_path):
    lines = open(file_path).readlines()
    random.shuffle(lines)

    with open(file_path, 'w') as f:
        f.writelines(lines)

def display_negative_percentage(sentence_num, num_negative, total_sentences):
    if total_sentences > 0:
        negative_percentage = (num_negative / sentence_num) * 100
        print(f"The percentage of negative sentences for sentence {sentence_num} is: {negative_percentage:.2f}% \n")
    else:
        print("No sentences to evaluate.")

def reading_sentence(input_file_path, output_file_path,result_file_path, target_type, alternative_type,max_sentences = 500):
    sentence_count = 0
    matching_sentence_count = 0

    with open(os.path.join(BASE_DIR, "TestComputer", input_file_path), "r") as input_file, open(
            os.path.join(BASE_DIR, "TestComputer", "Result", output_file_path), "w") as output_file, open(
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
            display_negative_percentage(sentence_count, matching_sentence_count, sentence_count)

            if "negative" in response:
                matching_sentence_count += 1
            else:
                error_message = f"Error sentence: {sentence_count}. {target_type} ==> {alternative_type}"
                output_file.write(f"{error_message}\n")
                output_file.write(f"{line}\n")
                output_file.write(f"{predict_class(clean_up_sentence(sentence), model)}\n")
                # bag = [1 if word in sentence else 0 for word in words]
                # output_file.write("\n".join(f"{x}: {word}" for x, word in zip(bag, sentence)) + "\n")
            if response == f"1001":
                break
            if sentence_count % 5 == 0:
                os.system("cls")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
            result_file.write(f"{matching_sentence_count}, {sentence_count - matching_sentence_count} \n")
    # return exactly result

    return [sentence_count, matching_sentence_count]





if __name__ == "__main__":
    os.system("cls")
    # shuffle_lines('TestComputer/Result/positive_v1.txt')
    # shuffle_lines('TestComputer/Result/negative_v1.txt')
    list1 = reading_sentence("Computer_posi.txt", "Computer_posi_ERROR.txt", "result_pos_ITEAI.txt", "positive", "negative")
    # os.system("cls")
    list2 = reading_sentence("Computer_nega.txt", "Computer_nega_ERROR.txt", "result_neg_ITEAI.txt", "negative", "positive")
    # os.system("cls")

    print(list1, list2)
    i = list1[0]
    j = list2[0]
    num_p = list1[1]
    num_n = list2[1]
    # Accuracy = tổng số câu đúng / tổng số câu hiện có
    total = i + j - 2
    accuracy = float((num_n + num_p) / (total))
    print('The number of positive sentences is: ' + str(num_p) + '\nThe number of negative sentences is: ' + str(num_n))
    print('\nTotal of number : ' + str(total))
    print('\nRate of exactly sentence : ' + str(100 * accuracy) + ' %')

    # # shuffle_lines('TestComputer/Result/result_pos.txt')
    toc = time.perf_counter()  # End Time

    # print(f"Build finished in {(toc - tic) / 60:0.0f} minutes {(toc - tic) % 60:0.0f} seconds")
    print(f"Build finished in {toc - tic:0.4f} seconds")

