# -*- coding:utf-8 -*-

import numpy as np
import json
import random
from keras.models import Sequential, load_model
from keras.layers import Dense,Dropout
from keras.optimizers import Adam
import pickle
import tensorflow as tf
global graph, sess
import nlp


graph = tf.Graph()
sess = tf.compat.v1.Session(graph = graph)

with open("intents.json", encoding='utf-8') as file:
    data = json.load(file)

with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
          
try:
    with graph.as_default():
        with sess.as_default():
            model = load_model('model.keras')  
    
except:    
    model = Sequential()
    model.add(Dense(16,input_shape=(len(training[0]),),activation="relu"))
    model.add(Dense(16,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(10,activation="softmax"))
    model.summary()
    model.compile(Adam(lr=.001),loss="categorical_crossentropy",metrics=["accuracy"])
    model.fit(training, output,epochs=200, verbose=2,batch_size=4)
    model.save("model.keras")


def prediction(x):
    with graph.as_default():
        with sess.as_default():
            results = model.predict(np.asanyarray([nlp.bag_of_words(x, words)]))[0]
            results_index = np.argmax(results)
            tag = labels[results_index]
            
            if results[results_index] > 0.70:
                for tg in data["intents"]:
                    if tg['tag'] == tag:
                        responses = tg['responses']
                return random.choice(responses)
            else:
                return "Tam olarak anlayamadım"
