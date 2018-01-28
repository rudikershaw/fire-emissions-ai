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

def train_validate_test_print(model, train_x, train_y):
    model.fit(
        train_x,
        train_y,
        epochs=40,
        validation_data=(
            np.genfromtxt('output/validation-features.csv', delimiter=','),
            np.genfromtxt('output/validation-targets.csv', delimiter=',')
        )
    )

    test_x = np.genfromtxt('output/test-features.csv', delimiter=',')
    test_y = np.genfromtxt('output/test-targets.csv', delimiter=',')

    print("\nTest evaluation: " + str(model.evaluate(test_x, test_y)) + "\n")

    print("\nPredicted: ")
    predictions = model.predict(test_x)
    print(predictions)

    print("\nExpected: ")
    print(test_y[0:3])
    print("...,")
    print(test_y[len(test_y)-3:len(test_y)])

    np.savetxt('output/test-predictions.csv', predictions, delimiter=',')

if __name__ == "__main__":
    TRAIN_X = np.genfromtxt('output/train-features.csv', delimiter=',')
    TRAIN_Y = np.genfromtxt('output/train-targets.csv', delimiter=',')

    MODEL = construct_model(TRAIN_X.shape[1], TRAIN_Y.shape[1])
    train_validate_test_print(MODEL, TRAIN_X, TRAIN_Y)
