from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD

import numpy as np
import pandas as pd

from sklearn import cross_validation

#read data
print('reading data...')
data = pd.read_csv('data/training_set.tsv', sep = '\t' )
ck12_features = pd.read_csv('features_ck12.csv', sep = ',' )
glove_features = pd.read_csv('features_glove.csv', sep = ',' )

dic={'A':[1,0,0,0], 'B':[0,1,0,0], 'C':[0,0,1,0], 'D':[0,0,0,1]}
#dic={'A': 0, 'B':1, 'C':2, 'D':3}

X_all = []
y_all = []
for index, row in data.iterrows():
    y_all.append(dic[row['correctAnswer']])
    X_all.append([glove_features['fA'][index], glove_features['fB'][index], glove_features['fC'][index], glove_features['fD'][index], 
                  ck12_features['fA'][index], ck12_features['fB'][index], ck12_features['fC'][index], ck12_features['fD'][index]])

X_all = np.array(X_all)
y_all = np.array(y_all)

X_train, X_test, y_train, y_test = cross_validation.train_test_split(X_all, y_all, test_size=0.3, random_state=0)

#create a model
print('creating model...')

model = Sequential()

model.add(Dense(32, input_dim=8, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='sgd', class_mode="categorical")

model.fit(X_train, y_train,
          nb_epoch=3,
          batch_size=16,
          show_accuracy=True, verbose=2)
#score = model.evaluate(X_test, y_test, batch_size=16)