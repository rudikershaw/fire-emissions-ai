import keras
import numpy as np
import tensorflow as tf
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers import Dense, Activation

features_train = np.genfromtxt('output/train-features.csv', delimiter=',')
targets_train = np.genfromtxt('output/train-targets.csv', delimiter=',')

features_test = np.genfromtxt('output/test-features.csv', delimiter=',')
targets_test = np.genfromtxt('output/test-targets.csv', delimiter=',')

model = Sequential()
model.add(Dense(units=116, activation='sigmoid', input_dim=features_train.shape[1]))
model.add(Dense(units=116, activation='sigmoid'))
model.add(Dense(units=targets_train.shape[1]))

model.compile(loss='mean_absolute_error', optimizer='sgd', metrics=['accuracy'])
model.fit(
    features_train,
    targets_train,
    epochs=2,
    validation_data=(features_test, targets_test)
)

print(model.predict(features_test))
