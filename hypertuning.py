import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import keras_tuner as kt

# The code is strongly inspired by
# https://towardsdatascience.com/hyperparameter-tuning-with-kerastuner-and-tensorflow-c4a4d690b31a

np.random.seed(42)


def build_model(hp):
    # Start model
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(17,)))

    # Number of hidden layers: 2 - 5
    # Number of units: 10-100 in steps of 10
    for i in range(1, hp.Int("num_layers", 2, 5)):
        model.add(
            keras.layers.Dense(
                units=hp.Int("units_" + str(i), min_value=10, max_value=100, step=10),
                activation=hp.Choice("activation_" + str(i), values=['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear', 'elu', 'selu']))
            )

        # Tune dropout layer with values from 0 - 0.3 with stepsize of 0.1.
        model.add(keras.layers.Dropout(hp.Float("dropout_" + str(i), 0, 0.3, step=0.1)))

        # Add output layer.
    model.add(keras.layers.Dense(units=10, activation="softmax"))

    # Tune learning rate for Adam optimizer with values from 0.01, 0.001, or 0.0001
    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    # Define optimizer, loss, and metrics
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model


tuner = kt.Hyperband(build_model,
                     objective="val_accuracy",
                     max_epochs=20,
                     factor=3,
                     hyperband_iterations=10,
                     directory="kt_dir",
                     project_name="kt_hyperband",)

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)


f = pd.read_csv("data/heart_2020_Female_encoded.csv")
m = pd.read_csv("data/heart_2020_Male_encoded.csv")

for data in (f, m):
    data = data.drop(columns="Unnamed: 0")
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    tuner.search(X_train, y_train, epochs=20, validation_data=(X_val, y_val),
                 callbacks=[stop_early], verbose=2)

    best_hps = tuner.get_best_hyperparameters()[0]
    print(best_hps)
