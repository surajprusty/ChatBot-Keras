import pickle
import random
import json
import numpy as np
from keras.models import load_model
from nltk_util import BOW

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

words=pickle.load(open('words.pkl','rb'))
classes=pickle.load(open('classes.pkl','rb'))
model=load_model('chatbotmodel.h5')


def predict_class(sentence):
    bow=BOW(sentence,words)
    res=model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD=0.25
    #print(res)
    results=[[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda  x: x[1], reverse=True)
    #print(results)
    return_list=[]

    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability':str(r[1])})

    return return_list

def get_response(intents_list,intent_json):
    tag=intents_list[0]['intent']
    prob=float(intents_list[0]['probability'])
    list_of_intents=intent_json['intents']
    #print(list_of_intents)
    if prob>0.75:
        for i in list_of_intents:
            if i['tag']==tag:
                result=random.choice(i['responses'])
                break
        print(f"{bot_name}: {result}")
        #print(f"{bot_name}: {result} :{prob}")
        #return result
    else:
        print(f"{bot_name}: I do not understand...")

bot_name = "Sam"
print("Let's chat! (type 'quit' to exit)")
while True:
    # sentence = "do you use credit cards?"
    sentence = input("You: ")
    if sentence == "quit":
        break

    ints=predict_class(sentence)
    res=get_response(ints,intents)
