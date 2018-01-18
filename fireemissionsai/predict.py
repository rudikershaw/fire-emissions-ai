import numpy as np
import tensorflow as tf

features = [
    tf.feature_column.numeric_column('x', shape=[227])
    # tf.feature_column.numeric_column('year', shape=[1]),
    # tf.feature_column.numeric_column('month', shape=[1]),
    # tf.feature_column.numeric_column('lon', shape=[1, 25]),
    # tf.feature_column.numeric_column('lat', shape=[1, 25]),
    # tf.feature_column.numeric_column('region', shape=[1, 25]),
    # tf.feature_column.numeric_column('BB', shape=[1, 25]),
    # tf.feature_column.numeric_column('NPP', shape=[1, 25]),
    # tf.feature_column.numeric_column('Rh', shape=[1, 25]),
    # tf.feature_column.numeric_column('C', shape=[1, 25]),
    # tf.feature_column.numeric_column('DM', shape=[1, 25]),
    # tf.feature_column.numeric_column('burned', shape=[1, 25])
]

estimator = tf.estimator.DNNRegressor(
    feature_columns=features,
    hidden_units=[20, 20]
)

features_train = np.genfromtxt('output/train-features.csv', delimiter=',')
targets_train = np.genfromtxt('output/train-targets.csv', delimiter=',')
features_test = np.genfromtxt('output/test-features.csv', delimiter=',')
targets_test = np.genfromtxt('output/test-targets.csv', delimiter=',')

input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": features_train},
    targets_train,
    batch_size=4,
    num_epochs=None,
    shuffle=True
)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": features_train},
    targets_train,
    batch_size=4,
    num_epochs=1000,
    shuffle=False
)
eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    {"x": features_test},
    targets_test,
    batch_size=4,
    num_epochs=1000,
    shuffle=False
)

# DNNRegressor only seems to handle single dimensional regression. Look for alternatives.
# estimator.train(input_fn=input_fn, steps=1000)

# train_metrics = estimator.evaluate(input_fn=train_input_fn)
# eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
# print("train metrics: %r"% train_metrics)
# print("eval metrics: %r"% eval_metrics)
