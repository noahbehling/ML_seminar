import numpy as np
import os
os.environ['PYTHONHASHSEED']=str(42)
import pandas as pd
from tensorflow import keras
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline
from create_model import create_model
from tensorflow.compat.v1 import set_random_seed
from tensorflow.compat.v1 import disable_v2_behavior, disable_eager_execution
import random
from sklearn.utils import shuffle

random.seed(42)

np.random.seed(42)
set_random_seed(42)
disable_eager_execution()
disable_v2_behavior()


f = pd.read_csv("data/new_heart_2020_Female_encoded.csv")
m = pd.read_csv("data/new_heart_2020_Male_encoded.csv")

# params = {
#             "model__n_layers": [1, 2, 3, 4, 5],
#             "model__units": [50, 100],
#             "model__activation": ["selu"],  # Better performance than tanh and relu in previous scans
#             "model__Dropout": [.1],
#             "model__learning_rate": [.001, 0.005, .01],
# }
params = {
            "model__n_layers": [3, 4],
            "model__units": [100, 150],
            "model__activation": ["elu"],
            "model__l_2": [False],
            "model__learning_rate": [5e-6],
}


for i, data in enumerate([f, m]):
    if i == 0:
        sex = "f"
    else:
        sex = "m"

    # data = data.drop(columns=["Unnamed: 0"])
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()
    neg, pos = np.bincount(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    steps = [('over', RandomOverSampler(sampling_strategy="minority", random_state=42)),
             ('model', KerasClassifier(build_fn=create_model, epochs=15,
                                       batch_size=128, verbose=0))]
    pipeline = Pipeline(steps=steps)

    grid = GridSearchCV(estimator=pipeline, param_grid=params, n_jobs=-1, verbose=1,
                        scoring='recall')
    grid_result = grid.fit(X_train, y_train)
    print(grid.best_params_)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)
    oversample = RandomOverSampler(sampling_strategy="minority", random_state=42)
    X_over, y_over = oversample.fit_resample(X_train, y_train)
    X_over, y_over = shuffle(X_over, y_over, random_state=42)

    model = create_model(n_layers=grid.best_params_["model__n_layers"],
                         units=grid.best_params_["model__units"],
                         activation=grid.best_params_["model__activation"],
                         l_2=grid.best_params_["model__l_2"],
                         learning_rate=grid.best_params_["model__learning_rate"])
    history = model.fit(x=X_over, y=y_over, validation_data=(X_val, y_val), epochs=50,
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

    plt.plot(x, history.history["precision"], label="Training")
    plt.plot(x, history.history["val_precision"], "--", label="Validation")
    plt.ylim(0, 1)
    plt.xlabel("Epcochs")
    plt.ylabel("Precision")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/precision_{sex}.pdf")
    plt.clf()

    plt.plot(x, history.history["accuracy"], label="Training")
    plt.plot(x, history.history["val_accuracy"], "--", label="Validation")
    plt.ylim(0, 1)
    plt.xlabel("Epcochs")
    plt.ylabel("Accuracy")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/accuracy_{sex}.pdf")
    plt.clf()

    plt.plot(x, history.history["loss"], label="Training")
    plt.plot(x, history.history["val_loss"], "--", label="Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Binary Crossentropy Loss")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/sklearn_loss_{sex}.pdf")
    plt.clf()

    keras.models.save_model(model, f"data/model_{sex}.tf", save_format="tf")
