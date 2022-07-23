import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# from pandas.plotting import scatter_matrix
import os
f = pd.read_csv("data/heart_2020_Female_encoded.csv")
m = pd.read_csv("data/heart_2020_Male_encoded.csv")

print(f['GenHealth'].median(), m['GenHealth'].median())
print(f['AgeCategory'].median(), m['AgeCategory'].median())

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
        if attr == "BMI":
            plt.axvline(30, 0, 1, ls="--", c="gray", label="Cut by hand")
        if attr == "Diabetic":
            plt.axvline(2.5, 0, 1, ls="--", c="gray", label="Cut by hand")
        if attr == "DiffWalking":
            plt.axvline(.5, 0, 1, ls="--", c="gray", label="Cut by hand")
        if attr == "AgeCategory":
            plt.axvline(6, 0, 1, ls="--", c="gray", label="Cut by hand")
        if attr == "Stroke":
            plt.axvline(.5, 0, 1, ls="--", c="gray", label="Cut by hand")
        if attr == "GenHealth":
            plt.axvline(2, 0, 1, ls="--", c="gray", label="Cut by hand")
        plt.legend(loc=0)
        plt.xlabel(f"{attr}")
        plt.ylabel("Density")
        plt.tight_layout()
        plt.savefig(f"plots/dists/{sex}/{attr}.pdf")
        plt.close()
    # scatter_matrix(data.drop(columns="HeartDisease"), c=data["HeartDisease"], figsize=(20,20))
    # plt.savefig(f"plots/dists/{sex}/scatter.pdf")


#create df to store results in
results = pd.DataFrame(columns = ['total', 'with hd', 'without hd', 'true positives', 'false positives', 'true negatives', 'false negatives'],
)

for f in (f, m):
    f_cut = f[(f['Stroke'] == 1) & (f['PhysicalHealth'] > 0) & (f['DiffWalking'] == 1) & (f['AgeCategory'] >= 7) & (f['Diabetic'] == 3) & (f['GenHealth'] <= 3)]
    f_nohd = pd.merge(f, f_cut, how='outer', indicator=True)
    f_nohd = f_nohd.query('_merge == "left_only"')
    f_nohd = f_nohd.drop(columns=['_merge'])

    f_results = {'total': len(f), 'with hd': f['HeartDisease'].sum(), 'without hd': len(f)-f['HeartDisease'].sum(),
                'true positives': f_cut['HeartDisease'].sum(), 'false positives': len(f_cut)-f_cut['HeartDisease'].sum(),
                'true negatives': len(f_nohd) - f_nohd['HeartDisease'].sum(), 'false negatives':f_nohd['HeartDisease'].sum()}

    results = results.append(f_results, ignore_index=True)

results['accuracy'] = (results['true positives'] + results['true negatives']) / results['total']
results['precision'] = results['true positives'] / (results['true positives'] + results['false positives'])
results['recall'] = results['true positives'] / (results['true positives'] + results['false negatives'])
results['f1'] = results['precision'] * results['recall'] / (results['precision'] + results['recall'])
print(results)
results.to_csv('manual_cut_results.csv')