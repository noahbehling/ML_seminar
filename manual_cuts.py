import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
import os

f = pd.read_csv("data/heart_2020_Female_encoded.csv")
m = pd.read_csv("data/heart_2020_Male_encoded.csv")

for i, data in enumerate([f, m]):
    if i == 0:
        sex = "f"
    else:
        sex = "m"
    if not os.path.exists("plots/dists"):
        os.makedirs("plots/dists")
        print("Created directory plots/dists")
    if not os.path.exists(f"plots/dists/{sex}"):
        os.makedirs(f"plots/dists/{sex}")
        print(f"Created directory plots/dists/{sex}")

    data = data.drop(columns=["Sex", "Unnamed: 0"])
    for attr in data.keys():
        if attr == "HeartDisease":
            continue
        data[attr][data["HeartDisease"] == 0].hist(alpha=.8, weights=np.full(data[data["HeartDisease"] == 0].shape[0], 1/data.shape[0]), label="No Heart Disease")
        data[attr][data["HeartDisease"] == 1].hist(alpha=.8, weights=np.full(data[data["HeartDisease"] == 1].shape[0], 1/data.shape[0]), label="Heart Disease")
        plt.legend(loc=0)
        plt.xlabel(f"{attr}")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"plots/dists/{sex}/{attr}.pdf")
        plt.close()
    # scatter_matrix(data.drop(columns="HeartDisease"), c=data["HeartDisease"], figsize=(20,20))
    # plt.savefig(f"plots/dists/{sex}/scatter.pdf")
