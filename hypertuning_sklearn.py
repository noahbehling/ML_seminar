import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.compat.v1 import set_random_seed

np.random.seed(42)
set_random_seed(42)


def FindLayerNodesLinear(n_layers, first_layer_nodes, last_layer_nodes):
    layers = []

    nodes_increment = (last_layer_nodes - first_layer_nodes) / (n_layers-1)
    nodes = first_layer_nodes
    for i in range(1, n_layers+1):
        layers.append(math.ceil(nodes))
        nodes = nodes + nodes_increment

    return layers


# Function to create model, required for KerasClassifier
def create_model(n_layers, units_0, activation_0, Dropout_0, units_1, activation_1,
                 Dropout_1, units_2, activation_2, Dropout_2, learning_rate):
    # create model
    model = Sequential()
    for i in range(n_layers):
        model.add(keras.layers.Flatten(input_shape=(17,)))
        model.add(keras.layers.Dense(units=eval(f"units_{i}"),
                                     activation=eval(f"activation_{i}")))
    model.add(keras.layers.Dropout(eval(f"Dropout_{i}")))

    model.add(keras.layers.Dense(units=1, activation='sigmoid'))
    # Compile model
    model.compile(loss=keras.losses.BinaryCrossentropy(),
                  optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=[
                        keras.metrics.TruePositives(name='tp'),
                        keras.metrics.FalsePositives(name='fp'),
                        keras.metrics.TrueNegatives(name='tn'),
                        keras.metrics.FalseNegatives(name='fn'),
                        keras.metrics.BinaryAccuracy(name='accuracy'),
                        keras.metrics.Precision(name='precision'),
                        keras.metrics.Recall(name='recall'),
                        keras.metrics.AUC(name='auc'),
                        keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
                  ])
    return model


f = pd.read_csv("data/heart_2020_Female_encoded.csv")
m = pd.read_csv("data/heart_2020_Male_encoded.csv")

params = {
            "n_layers": [1, 2, 3],
            "units_0": np.linspace(10, 100, 5),
            "activation_0": ["relu", "tanh", "selu"],
            "Dropout_0": [.3, .2],
            "learning_rate": np.linspace(1e-3, 1e-2, 5),
            "units_1": np.linspace(10, 100, 5),
            "activation_1": ["relu", "tanh", "selu"],
            "Dropout_1": [.3, .2],
            "units_2": np.linspace(10, 100, 5),
            "activation_2": ["relu", "tanh", "selu"],
            "Dropout_2": [.3, .2]
}


for i, data in enumerate([f, m]):
    data = data.drop(columns="Unnamed: 0")
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()
    neg, pos = np.bincount(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    bias = np.log(pos / neg)

    model = KerasClassifier(build_fn=create_model, epochs=10,
                             batch_size=128, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, verbose=1,
                        scoring='recall')
    grid_result = grid.fit(X_train, y_train)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    model = create_model(params=grid.best_params_, metrics=[
          keras.metrics.TruePositives(name='tp'),
          keras.metrics.FalsePositives(name='fp'),
          keras.metrics.TrueNegatives(name='tn'),
          keras.metrics.FalseNegatives(name='fn'),
          keras.metrics.BinaryAccuracy(name='accuracy'),
          keras.metrics.Precision(name='precision'),
          keras.metrics.Recall(name='recall'),
          keras.metrics.AUC(name='auc'),
          keras.metrics.AUC(name='prc', curve='PR'),  # precision-recall curve
    ], output_bias=bias)
    model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=30,
              verbose=2,
              batch_size=128)


    if i == 0:
        model.save("data/model_f.tf")
    else:
        model.save("data/model_m.tf")
