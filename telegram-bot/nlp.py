from snowballstemmer import TurkishStemmer
import json
import pickle
import nltk
import numpy as np


with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

stemmer=TurkishStemmer()
    
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
     

except:                
    
    words = []
    labels = []
    docs_x = []
    docs_y = []
    
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            
    words = [stemmer.stemWord(w.lower()) for w in words if w != "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)
                                                          
    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]
    
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stemWord(w.lower()) for w in doc]
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1
        training.append(bag)
        output.append(output_row)
        
    training = np.array(training)
    output = np.array(output)
    
    with open("data.pickle", "wb") as f:
            pickle.dump((words, labels, training, output), f)
     
        
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stemWord(word.lower()) for word in s_words]
    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
     
    return np.array(bag)