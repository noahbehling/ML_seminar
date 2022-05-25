import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.viridis()


def produce_plots(a):
    colours = {0: 'green', 1:'red'}
    if a == 'f':
        f = pd.read_csv('data/heart_2020_Female_encoded.csv', index_col=0)
        f = f.drop(columns='Sex')
        print('read in f')
        plt.figure(figsize=(30, 30))
        sns.heatmap(f.corr(), annot=True, square=True, cmap='coolwarm')
        plt.savefig('plots/f_heatmap.pdf')

        f.hist(alpha=0.8, figsize=(20, 20))
        plt.savefig('plots/f_hist.pdf')

        pd.plotting.scatter_matrix(f.drop(columns = ['HeartDisease', 'GenHealth', 'DiffWalking', 'Asthma', 'KidneyDisease', 'Smoking', 'AlcoholDrinking', 'SkinCancer', 'Stroke', 'PhysicalActivity']), c=f['HeartDisease'].map(colours) ,alpha=0.8, figsize=(30, 30), s=20)
        plt.savefig('plots/f_scatter.pdf')

    if a == 'm':
        m = pd.read_csv('data/heart_2020_Male_encoded.csv', index_col=0)
        m = m.drop(columns='Sex')
        print('read in m')
        plt.figure(figsize=(30, 30))
        sns.heatmap(m.corr(), annot=True, square=True, cmap='coolwarm')
        plt.savefig('plots/m_heatmap.pdf')

        m.hist(alpha=0.8, figsize=(20, 20))
        plt.savefig('plots/m_hist.pdf')

produce_plots('f')
produce_plots('m')