import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import numpy as np

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

    print('test data accuracy', reg.score(X_test, y_test))
    #print('train data accuracy', reg.score(X_train, y_train))
    print(confusion_matrix(y_test, y_pred))