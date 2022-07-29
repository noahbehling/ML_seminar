import os
os.environ['PYTHONHASHSEED']=str(42)
import numpy as np
import pandas as pd
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import shap
from tensorflow.compat.v1 import set_random_seed
from tensorflow.compat.v1 import disable_v2_behavior, disable_eager_execution
from create_model import create_model
import random

random.seed(42)
np.random.seed(42)
set_random_seed(42)
disable_eager_execution()
disable_v2_behavior()


f = pd.read_csv("data/new_heart_2020_Female_encoded.csv")
m = pd.read_csv("data/new_heart_2020_Male_encoded.csv")
lin = pd.read_csv("manual_cut_results.csv")

for i, data in enumerate([f, m]):
    if i == 0:
        sex = "f"
    else:
        sex = "m"

    # data = data.drop(columns=["Unnamed: 0", "Sex"])
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    model = keras.models.load_model(f"data/model_{sex}.tf")
    model.summary()
    print(model.evaluate(X_test, y_test))
    y_pred = model.predict(X_test, batch_size=128)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    plt.figure(figsize=(6.4, 4.8))
    plt.plot(fpr, tpr, label="ROC curve")
    if i == 0:
        tp = 9334
        fp = 70488
        tn = 86083
        fn = 1900
    else:
        tp = 13444
        fp = 52739
        tn = 83112
        fn = 2695
    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)
    plt.plot(fpr, tpr, "y+", label="Linear cuts, corr >0.25")
    tp = lin["true positives"][i]
    fp = lin["false positives"][i]
    tn = lin["true negatives"][i]
    fn = lin["false negatives"][i]
    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)
    ln = np.linspace(0, 1, 1000)
    plt.plot(ln, ln, "r--", label="Random guessing")
    plt.plot(fpr, tpr, "k+", label="Linear cuts, corr >0.20")
    if i == 0:
        tp = 387
        fp = 401
        tn = 156170
        fn = 10847
    else:
        tp = 385
        fp = 219
        tn = 135632
        fn = 15754
    fpr = fp/(fp + tn)
    tpr = tp/(tp + fn)
    plt.plot(fpr, tpr, "g+", label="Linear cuts, corr >0.15")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc=0)
    plt.tight_layout()
    plt.savefig(f"plots/test_roc_{sex}.pdf")
    plt.clf()

    y_cls = (y_pred > .5).astype("int32")
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
    plt.xlabel('Probability of being labelled as person with heart disease')
    plt.ylabel('Number of entries')
    plt.savefig(f"plots/prob_{sex}.pdf")
    plt.clf()

    explainer = shap.DeepExplainer(model, X_train[:1000, :])
    shap_values = explainer.shap_values(X_test[100:1000, :], check_additivity=True
                                        )

    shap.summary_plot(shap_values[0], X_test[100:1000], show=False,
                      feature_names=data.drop(columns=["HeartDisease"]
                                              ).columns.tolist(),
                      plot_type="bar")
    plt.savefig(f"plots/feature_importance_{sex}.pdf")
    plt.clf()

    shap.summary_plot(shap_values[0], X_test[100:1000], show=False,
                      feature_names=data.drop(columns=["HeartDisease"]
                                              ).columns.tolist(),
                      )
    plt.savefig(f"plots/understanding_performance_{sex}.pdf")
    plt.clf()
