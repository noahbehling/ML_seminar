import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt

#load data, perform train test split
f = pd.read_csv("data/heart_2020_Female_encoded.csv")
m = pd.read_csv("data/heart_2020_Male_encoded.csv")

i = 0
for data in (f, m):
    data = data.drop(columns="Unnamed: 0")
    X = data.drop(columns="HeartDisease").to_numpy()
    y = data["HeartDisease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,
                                                        random_state=42)

    reg = linear_model.LogisticRegression(max_iter = 1000)
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    lr_probs = reg.predict_proba(X_test)
    lr_probs = lr_probs[:, 1]
    ns_probs = [0 for i in range(len(y_test))]
    ns_auc = roc_auc_score(y_test, ns_probs)
    lr_auc = roc_auc_score(y_test, lr_probs)

    ns_fpr, ns_tpr, ns_ = roc_curve(y_test, ns_probs)
    lr_fpr, lr_tpr, lr_ = roc_curve(y_test, lr_probs)

    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))

    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', markersize=0.05, label='Logistic Regression')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()

    if i == 0:
        plt.savefig('plots/rocauc_f.pdf')
    else:
        plt.savefig('plots/rocauc_m.pdf')

    i +=1
    plt.close()

    print('test data accuracy', reg.score(X_test, y_test))
    #print('train data accuracy', reg.score(X_train, y_train))
    print(confusion_matrix(y_test, y_pred))