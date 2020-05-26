#!/usr/bin/env python
# coding: utf-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from keras.datasets import mnist
import numpy as np
dataset = mnist.load_data('mnist.db')

train , test = dataset

import sys 

X_train , y_train =train

#here 6000 is number of records and the size of each record is 2 dimmensional
#and the size is 28*28

img1 = X_train[0]

X_test ,y_test = test

img1_label = y_train[0]


X_train_1d = X_train.reshape(-1 , 28*28)
X_test_1d = X_test.reshape(-1 , 28*28)


from keras.utils.np_utils import to_categorical

y_train_cat = to_categorical(y_train)

X_train = X_train_1d.astype('float32')
X_test = X_test_1d.astype('float32')


from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

unit = int(sys.argv[1])
epoch = int(sys.argv[2])

model.add(Dense(units=unit, input_dim=28*28, activation='relu'))

unit = int(unit/2) 

model.add(Dense(units=unit, activation='relu'))

unit = int(unit/2)

model.add(Dense(units=unit, activation='relu'))

unit = int(unit/2)

model.add(Dense(units=unit, activation='relu'))

model.add(Dense(units=10, activation='softmax'))

from keras.optimizers import Adam

model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', 
             metrics=['accuracy']
             )

h = model.fit(X_train, y_train_cat, epochs=epoch)
acc = h.history['accuracy']
model.save("numberpredictionmodel.pk1")


file = open("accuracy_score.txt", "w") 

accuracy = str(acc[-1]*100)

file.write(accuracy)

file.close()