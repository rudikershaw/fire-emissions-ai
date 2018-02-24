"""The predict module is for taking a .csv of month specific fire emissions features
and predicting the values for the following month. All the predict.py command requires
path to such a .csv file.

Optionally providing a --retrain parameter will have the predict.py command attempt to
retrain its model on data in the ouput folder (to get data for training, use the
preprocess.py module) before predicting values with the new model. If the --persist
parameter is also provided the new learned model and its weights will be persisted for
future use.
"""

import os
import keras
import numpy as np
from keras.models import load_model
from keras.layers import Dense, BatchNormalization
from argparse import ArgumentParser, RawTextHelpFormatter

def construct_model(input_shape, output_shape):
    """Construct the model for predicting next month's fire emissions data."""
    model = keras.models.Sequential()
    model.add(Dense(units=412, activation='relu', input_dim=input_shape))
    model.add(BatchNormalization())
    model.add(Dense(units=412, activation='relu'))
    model.add(Dense(units=412, activation='relu'))
    model.add(Dense(units=output_shape, activation='relu'))

    sgd = keras.optimizers.SGD(lr=0.02, momentum=0.8, decay=1e-6)
    model.compile(loss='mean_absolute_error', optimizer=sgd, metrics=['accuracy'])
    return model

def train_validate_test_print(model, train_x, train_y, input_x, persist):
    """Train a provided model on the provided training set."""
    model.fit(
        train_x,
        train_y,
        epochs=20,
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
    np.savetxt('output/predictions.csv', model.predict(input_x), delimiter=',')
    if persist:
        model.save('model_weights.h5')

if __name__ == "__main__":
    PARSER = ArgumentParser(description=__doc__, formatter_class=RawTextHelpFormatter)
    PARSER.add_argument("inputs", help="Path to .csv with input features for prediction")
    PARSER.add_argument("--retrain", dest='retrain', action='store_true', help='Retrain the model')
    PARSER.add_argument("--persist", dest='persist', action='store_true', help='Save retrained model')
    PARSER.add_argument("--debug", dest='debug', action='store_true')
    PARSER.set_defaults(retrain=False)
    PARSER.set_defaults(persist=False)
    PARSER.set_defaults(debug=False)
    ARGS = PARSER.parse_args()

    if not ARGS.debug:
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    INPUT_X = np.genfromtxt(ARGS.inputs, delimiter=',')
    if ARGS.retrain:
        TRAIN_X = np.genfromtxt('output/train-features.csv', delimiter=',')
        TRAIN_Y = np.genfromtxt('output/train-targets.csv', delimiter=',')

        MODEL = construct_model(TRAIN_X.shape[1], TRAIN_Y.shape[1])
        train_validate_test_print(MODEL, TRAIN_X, TRAIN_Y, INPUT_X, ARGS.persist)
    else:
        MODEL = load_model('model_weights.h5')
        np.savetxt('output/predictions.csv', MODEL.predict(INPUT_X), delimiter=',')
        print('\nPredictions saved to output directory.\n')
