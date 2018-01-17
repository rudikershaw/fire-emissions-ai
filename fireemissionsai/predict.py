import numpy as np
import tensorflow as tf

features = [
    tf.feature_column.numeric_column('year', shape=[1]),
    tf.feature_column.numeric_column('month', shape=[1]),
    tf.feature_column.numeric_column('lon', shape=[1, 25]),
    tf.feature_column.numeric_column('lat', shape=[1, 25]),
    tf.feature_column.numeric_column('region', shape=[1, 25]),
    tf.feature_column.numeric_column('BB', shape=[1, 25]),
    tf.feature_column.numeric_column('NPP', shape=[1, 25]),
    tf.feature_column.numeric_column('Rh', shape=[1, 25]),
    tf.feature_column.numeric_column('C', shape=[1, 25]),
    tf.feature_column.numeric_column('DM', shape=[1, 25]),
    tf.feature_column.numeric_column('burned', shape=[1, 25]),
]

estimator = tf.estimator.DNNRegressor(
    feature_columns=features,
    hidden_units=[256, 512, 256],
    optimizer=tf.train.ProximalAdagradOptimizer(
        learning_rate=0.1,
       l1_regularization_strength=0.001
    )
)

features_train = np.genfromtxt('output/train-features.csv', delimiter=',')
targets_train = np.genfromtxt('output/train-targets.csv', delimiter=',')
features_test = np.genfromtxt('output/test-features.csv', delimiter=',')
targets_test = np.genfromtxt('output/test-targets.csv', delimiter=',')
