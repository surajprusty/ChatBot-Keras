import nltk
import pickle
import numpy as np
#nltk.download('punkt')
from nltk.stem import WordNetLemmatizer

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word.lower())

def BOW(sentences,words):
    sentences=tokenize(sentences)
    sentence_words = [stem(word) for word in sentences]
    bag=[0]*len(words)
    for w in sentence_words:
        for i,word in enumerate(words):
            if word==w:
                bag[i]=1
    return np.array(bag)