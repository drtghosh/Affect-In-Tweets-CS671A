# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 12:34:15 2018

@author: Debabrata
"""

import numpy
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.datasets import imdb
import sys

orig_stdout = sys.stdout
f = open('BBoW_out_LSTM.txt', 'w')
sys.stdout = f

#Building Recurrent Neural Network Model
numpy.random.seed(0)
main_features = 20000
#Getting train and test data in binary BOW form for RNN as given in Keras documentation
(x_train, y_train), (x_test, y_test) = imdb.load_data(path="imdb.npz", num_words=main_features)
# padding or pruning of input sequences
max_len_review =250
X_train = sequence.pad_sequences(x_train, maxlen=max_len_review)
X_test = sequence.pad_sequences(x_test, maxlen=max_len_review)
length_embedding = 128
#construct the seq RNN
r_network = Sequential()
#Embedding
r_network.add(Embedding(main_features, length_embedding, input_length=max_len_review))
r_network.add(LSTM(60))
# Add fully connected layer with a sigmoid activation function
r_network.add(Dense(1, activation='sigmoid'))
# Compile neural network
r_network.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print(r_network.summary())


# Train neural network
r_network.fit(X_train, y_train, epochs=2, batch_size=30, verbose=0)

#Testing 
y_pred_LSTM = r_network.predict(X_test)
threshold = 0.5
targets = ['Negative', 'Positive']

print("RNN-LSTM: ",classification_report(y_test, y_pred_LSTM > threshold, target_names=targets))

sys.stdout = orig_stdout
f.close()