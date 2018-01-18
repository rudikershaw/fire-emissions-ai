import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

features_train = np.genfromtxt('output/train-features.csv', delimiter=',')
targets_train = np.genfromtxt('output/train-targets.csv', delimiter=',')

model = Sequential([
    Dense(100, input_shape=(227,)),
    Activation('relu'),
    Dense(6),
    Activation('softmax'),
])
# model.add(Dense(units=100, activation='relu', input_dim=227))
# model.add(Dense(units=6, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
model.fit(features_train, targets_train, epochs=10, batch_size=32)

features_test = np.genfromtxt('output/test-features.csv', delimiter=',')
targets_test = np.genfromtxt('output/test-targets.csv', delimiter=',')

loss_and_metrics = model.evaluate(features_test, targets_test, batch_size=128)
