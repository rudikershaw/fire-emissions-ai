import keras
import numpy as np
from keras.layers import Dense, BatchNormalization

def construct_model(input_shape, output_shape):
    model = keras.models.Sequential()
    model.add(Dense(units=112, activation='relu', input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(units=112, activation='relu'))
    model.add(Dense(units=output_shape, activation='relu'))

    sgd = keras.optimizers.SGD(lr=0.02, momentum=0.8, decay=1e-6)
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
    return model

if __name__ == "__main__":
    TRAIN_X = np.genfromtxt('output/train-features.csv', delimiter=',')
    TRAIN_Y = np.genfromtxt('output/train-targets.csv', delimiter=',')

    MODEL = construct_model(TRAIN_X.shape[1], TRAIN_Y.shape[1])
    MODEL.fit(
        TRAIN_X,
        TRAIN_Y,
        epochs=1,
        validation_data=(
            np.genfromtxt('output/validation-features.csv', delimiter=','),
            np.genfromtxt('output/validation-targets.csv', delimiter=',')
        )
    )

    TEST_X = np.genfromtxt('output/test-features.csv', delimiter=',')
    TEST_Y = np.genfromtxt('output/test-targets.csv', delimiter=',')

    print("\nTest evaluation: " + str(MODEL.evaluate(TEST_X, TEST_Y)) + "\n")

    print("\nPredicted: ")
    PREDICTIONS = MODEL.predict(TEST_X)
    print(PREDICTIONS)

    print("\nExpected: ")
    print(TEST_Y[0:3])
    print("...,")
    print(TEST_Y[len(TEST_Y)-3:len(TEST_Y)])

    np.savetxt('output/test-predictions.csv', PREDICTIONS, delimiter=',')
