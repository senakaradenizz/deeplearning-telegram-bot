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
import matplotlib.pyplot as plt


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
    model.compile(Adam(lr=.001),loss="categorical_crossentropy",metrics=["acc"])
    history = model.fit(training, output, validation_split = 0.1, epochs=200, verbose=1,batch_size=4)
    model.save("model.keras")
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']


epochs = range(1,len(acc)+1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()

plt.show()

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
