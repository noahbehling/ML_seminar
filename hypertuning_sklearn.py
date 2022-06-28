import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.compat.v1 import set_random_seed
import matplotlib.pyplot as plt
import joblib

np.random.seed(42)
set_random_seed(42)


# Function to create model, required for KerasClassifier
def create_model(n_layers, units, activation, Dropout, learning_rate):
    # create model
    model = Sequential()
    for i in range(n_layers):
        model.add(keras.layers.Flatten(input_shape=(17,)))
        model.add(keras.layers.Dense(units=eval("units"),
                                     activation=eval("activation")))
    model.add(keras.layers.Dropout(eval("Dropout")))

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
            "units": np.linspace(10, 100, 5),
            "activation": ["relu", "tanh", "selu"],
            "Dropout": [.1, .2],
            "learning_rate": np.linspace(1e-3, 1e-2, 5),
}
# params = {
#             "n_layers": [1],
#             "units": [10],
#             "activation": ["relu"],
#             "Dropout": [.1],
#             "learning_rate": [1e-3],
# }


for i, data in enumerate([f, m]):
    if i == 0:
        sex = "f"
    else:
        sex = "m"

    data = data.drop(columns="Unnamed: 0")
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()
    neg, pos = np.bincount(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    bias = np.log(pos / neg)

    model = KerasClassifier(build_fn=create_model, epochs=15,
                            batch_size=128, verbose=0)
    grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, verbose=1,
                        scoring='recall')
    grid_result = grid.fit(X_train, y_train)
    joblib.dump(grid, f"data/grid_{sex}.pkl")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    model = create_model(n_layers=grid.best_params_["n_layers"],
                         units=grid.best_params_["units"],
                         activation=grid.best_params_["activation"],
                         Dropout=grid.best_params_["Dropout"],
                         learning_rate=grid.best_params_["learning_rate"])
    history = model.fit(x=X_train, y=y_train, validation_data=(X_val, y_val), epochs=50,
                        verbose=2,
                        batch_size=128)

    x = np.arange(1, len(history.history["recall"])+1)
    plt.plot(x, history.history["recall"], label="Training")
    plt.plot(x, history.history["val_recall"], "--", label="Validation")
    plt.ylim(0, 1)
    plt.xlabel("Epcochs")
    plt.ylabel("Recall")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/recall_{sex}.pdf")
    plt.clf()

    plt.plot(x, history.history["loss"], label="Training")
    plt.plot(x, history.history["val_loss"], "--", label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/sklearn_loss_{sex}.pdf")
    plt.clf()

    model.save(f"data/model_{sex}.tf")
