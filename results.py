from tensorflow import keras
from keras.models import Sequential
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import joblib
import shap


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

for i, data in enumerate([f, m]):
    if i == 0:
        sex = "f"
    else:
        sex = "m"

    data = data.drop(columns="Unnamed: 0")
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    model = keras.models.load_model(f"data/model_{sex}.tf")
    grid = joblib.load(f"data/grid_{sex}.pkl")
    print(f"Best params for {sex}: ", grid.best_params_)
    print(model.evaluate(X_test, y_test))
    y_pred = model.predict(X_test, batch_size=128)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.plot(fpr, tpr, label="ROC curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/test_roc_{sex}.pdf")
    plt.clf()

    y_cls = (y_pred > .8).astype("int32")
    print(f"{sex} Classification Report: ", classification_report(y_test, y_cls))

    # Plot 0 probability including overtraining test
    plt.figure(figsize=(8, 8))

    label = 1
    # Test prediction
    plt.hist(y_pred[y_test == label], alpha=0.5, color='red', range=[0, 1], bins=10)
    plt.hist(y_pred[y_test != label], alpha=0.5, color='blue', range=[0, 1], bins=10)

    # Train prediction
    Y_train_pred = model.predict(X_train)
    plt.hist(Y_train_pred[y_train == label], alpha=0.5, color='red', range=[0, 1],
             bins=10, histtype='step', linewidth=2)
    plt.hist(Y_train_pred[y_train != label], alpha=0.5, color='blue', range=[0, 1],
             bins=10, histtype='step', linewidth=2)

    plt.legend(['train == 1', 'train == 0', 'test == 1', 'test == 0'], loc='upper right')
    plt.xlabel('Probability of having a heart disease')
    plt.ylabel('Number of entries')
    plt.savefig(f"plots/prob_{sex}.pdf")

    # explainer = shap.Explainer(model)
    # shap_values = explainer(X_test)
    # shap.plots.bar(shap_values)
    # plt.savefig(f"plots/feature_importance_{sex}.pdf")
