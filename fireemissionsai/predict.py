import keras
import numpy as np
from keras.layers import Dense, BatchNormalization

features_train = np.genfromtxt('output/train-features.csv', delimiter=',')
targets_train = np.genfromtxt('output/train-targets.csv', delimiter=',')

features_test = np.genfromtxt('output/test-features.csv', delimiter=',')
targets_test = np.genfromtxt('output/test-targets.csv', delimiter=',')

model = keras.models.Sequential()
model.add(Dense(units=112, input_dim=features_train.shape[1]))
model.add(BatchNormalization())
model.add(Dense(units=112))
model.add(Dense(units=targets_train.shape[1], activation='relu'))

sgd = keras.optimizers.SGD(lr=0.01, momentum=0.4, decay=1e-6)
model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
model.fit(
    features_train,
    targets_train,
    epochs=10,
    validation_data=(features_test, targets_test)
)

print("\nPredicted: ")
print(model.predict(features_test))

print("\nExpected: ")
print(targets_test[0:3])
print("...,")
print(targets_test[len(targets_test)-3:len(targets_test)])
