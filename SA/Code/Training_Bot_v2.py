import nltk
import os
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
import random
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

lemmatizer = WordNetLemmatizer()

# Base directory = thư mục nc/
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Đọc dữ liệu từ tệp JSON
data = json.loads(open(os.path.join(BASE_DIR, 'Data', 'intent', 'jon.json')).read())

words = []
classes = []
documents = []
ignore_words = ['?', '!']

for intent in data['intents']:
    for pattern in intent['patterns']:
        # Tokenize từng từ trong mẫu
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# sử dụng từ một cách duy nhất và loại bỏ các từ trùng lặp
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

# Lưu trữ dữ liệu đã xử lý
pickle.dump(words, open(os.path.join(BASE_DIR, 'Data', 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(BASE_DIR, 'Data', 'classes.pkl'), 'wb'))

training = []
output_empty = [0] * len(classes)

# Tạo dữ liệu huấn luyện từ các mẫu và phản hồi
for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

random.shuffle(training)

# Tạo mảng numpy tương thích với phiên bản NumPy mới
dims = (len(training), len(training[0]))
arr = np.empty(dims, dtype=object)
for i in range(len(training)):
    arr[i] = training[i]
training = np.array(arr)

train_x = list(training[:, 0])
train_y = list(training[:, 1])


# Tạo và biên dịch mô hình
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Sử dụng trình tạo optimizer của Keras 3
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Huấn luyện mô hình với dữ liệu mới
hist = model.fit(np.array(train_x), np.array(train_y), epochs=30, batch_size=2, verbose=1)

# Lưu mô hình và lịch sử
model.save(os.path.join(BASE_DIR, 'Code', 'Model', 'OWLChatModel_v4.h5'))
with open(os.path.join(BASE_DIR, 'Code', 'Model', 'history.pkl'), 'wb') as f:
    pickle.dump(hist.history, f)

print("suses")
print("Số từ đã được huấn luyện từ tệp JSON là:", len(words))

print("---------------------------------------------------------------")

# Lấy tất cả các từ từ văn bản gốc
# Lấy tất cả các từ từ các mẫu dữ liệu
all_words = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        all_words.extend(w)

all_words = [lemmatizer.lemmatize(word.lower()) for word in all_words if word not in ignore_words]

# Lấy các từ không được huấn luyện
not_trained_words = [word for word in all_words if word not in words]

print("Số từ không được huấn luyện là:", len(not_trained_words))
