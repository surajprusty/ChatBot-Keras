import pickle
import random
import numpy as np
import json
import string
from nltk_util import tokenize, stem
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dropout, Dense

words= []
classes=[]
documents=[]

with open('intents.json', 'r') as f:
    intents = json.load(f)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list=tokenize(pattern)
        for i in word_list:
            words.append(i)
            documents.append((word_list,intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

words= [stem(w.lower()) for w in words if w not in string.punctuation]
words= sorted(set(words))

pickle.dump(words, open('words.pkl','wb'))
pickle.dump(classes, open('classes.pkl','wb'))

training=[]
output_empty=[0]*len(classes)

for document in documents:
    bag=[]
    word_patterns=document[0]
    word_patterns=[stem(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row=list(output_empty)
    output_row[classes.index(document[1])]=1
    training.append([bag,output_row])

random.shuffle(training)
training=np.array(training)

train_x=list(training[:,0])
train_y=list(training[:,1])

###MODEL TRAINING###
model= Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0],), activation='softmax'))
#adam = keras.optimizers.Adam(learning_rate=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = keras.optimizers.Adam()
model.compile(loss='categorical_crossentropy',optimizer=adam, metrics=['accuracy'])
fmodel=model.fit(np.array(train_x),np.array(train_y), epochs=100, batch_size=5,verbose=True)
model.save('chatbotmodel.h5',fmodel)
print("Done")