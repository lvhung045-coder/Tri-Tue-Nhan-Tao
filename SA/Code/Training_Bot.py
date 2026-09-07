import nltk
import os
# nltk.download()
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

import json
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random

words = []
classes = []
documents = []
ignore_words = ['?', '!']

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_file = open(os.path.join(BASE_DIR, 'Data', 'intent', 'intents001.json')).read()
intents = json.loads(data_file)
print(words)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# lemmaztize and lower each word and remove duplicates
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
# sort classes
classes = sorted(list(set(classes)))
# documents = combination between patterns and intents
print(len(documents), "documents")
# classes = intents
print(len(classes), "classes", classes)
# words = all words, vocabulary
print(len(words), "unique lemmatized words", words)

pickle.dump(words, open(os.path.join(BASE_DIR, 'Data', 'words.pkl'), 'wb'))
pickle.dump(classes, open(os.path.join(BASE_DIR, 'Data', 'classes.pkl'), 'wb'))
# create our training data
training = []
# create an empty array for our output
output_empty = [0] * len(classes)
# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []
    # list of tokenized words for the pattern
    pattern_words = doc[0]
    # lemmatize each word - create base word, in attempt to represent related words
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    # create our bag of words array with 1, if word match found in current pattern
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    # output is a '0' for each tag and '1' for current tag (for each pattern)
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])
# shuffle our features and turn into np.array
random.shuffle(training)

# Get the size of the training list
dims = (len(training), len(training[0]))

# Create array of objects of similar size
arr = np.empty(dims, dtype=object)

# Assign each subarray to the object array
for i in range(len(training)):
   arr[i] = training[i]

# Switch back to array
training = np.array(arr)
# create train and test lists. X - patterns, Y - intents
train_x = list(training[:, 0])  # need to train larger
train_y = list(training[:, 1])
print("Training data created")

# Create model - 3 layers. First layer 128 neurons, second layer 64 neurons and 3rd output layer contains number of neurons
# equal to number of intents to predict output intent with softmax
model = Sequential()

model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.01))

model.add(Dense(64, activation='relu'))
model.add(Dropout(0.01))

model.add(Dense(len(train_y[0]), activation='softmax'))

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# fitting and saving the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=19, batch_size=5, verbose=1) 

# accuracy = [x * 100 for x in hist.history['accuracy']]
# loss = [y * 100 for y in hist.history['loss']]
# plt.plot(accuracy)
# plt.plot(loss)

# plt.title('Model Accuracy and Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Value')
# plt.legend(['Accuracy', 'Loss'], loc='upper left')
# plt.savefig('model_accuracy_loss.png')

model.save(os.path.join(BASE_DIR, 'Code', 'Model', 'OWLChatModel_v2.h5'), hist)
# model.save('Model/my_model.keras', hist)

print("model created")

