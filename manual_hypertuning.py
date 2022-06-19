import tensorflow as tf

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools

from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, InputLayer
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import tensorflow.keras as keras

np.random.seed(42)


def build_model(params):
    model = Sequential()
    model.add(keras.layers.Flatten(input_shape=(17,)))

    for layer in range(params['num_layers']):
        model.add(keras.layers.Dense(units = params['units'], activation=params['activation_function'])) #change to try out outher act fcts
        
        model.add(keras.layers.Dropout(params['dropout']))
    
    #add output layer
    model.add(keras.layers.Dense(units=10, activation="softmax"))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['learning_rate']),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=["accuracy"])

    return model





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
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=.2,
                                                      random_state=42)

    param_space = {
        'num_layers': [2, 3], #also tried: 5,1
        'units': [90, 100], #also tried: 10, 80
        'dropout': [0.3, 0.2], #also tried: 0.1, 0.2
        'dense_nodes': [32, 64],
        'activation_function': ['relu'],#'hard_sigmoid', 'linear'], also tried: hard_sigmoid, tanh, 'sigmoid', 'linear'
        'learning_rate': [1e-2, 1e-3] #also tried: 1e-4; 1e-3 is good as well
    }

    #create param combis
    value_combis = itertools.product(*[v for v in param_space.values()])
    param_combis = [{key:value for key, value in zip(param_space.keys(), combi)} for combi in value_combis]


    search_results = []

    k_folds = 3

    for idx, params in enumerate(param_combis):
        print(f"Start run {idx+1}/{len(param_combis)}: Parameters: {params}")
    
        k = 5
        kf = StratifiedKFold(n_splits=k)

        #append during the validation folds
        best_val_accs             = []
        best_val_acc_losses       = []
        best_val_acc_train_accs   = []
        best_val_acc_train_losses = []

        y_labels = y_train

        for k_index, (train_idx, val_idx) in enumerate(kf.split(X_train, y_labels)):
        
            x_cv_train, x_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]

            #create correct filepath
            filepath = f'./paramsearch/cnn_paramsearch_filters_fold={k_index}_'
            for key, value in params.items():
                filepath += key + '=' + str(value) + '_'
            filepath +='.hdf5'

            checkpoint = ModelCheckpoint(
            filepath, monitor='val_accuracy', verbose=0, save_best_only=True, mode='max'
            )

            this_model = build_model(params)

            fit_results = this_model.fit(
                x = x_cv_train,
                y = y_cv_train,
                batch_size = 128,
                epochs = 30, 
                validation_data = (x_cv_val, y_cv_val),
                callbacks = [checkpoint],
                verbose = 2
            )

            #extract best validation scores
            best_val_epoch = np.argmax(fit_results.history['val_accuracy'])
            best_val_accs.append(np.max(fit_results.history['val_accuracy']))
            best_val_acc_losses.append(fit_results.history['val_loss'][best_val_epoch])

            # get correct training accuracy
            best_model = load_model(filepath)
            best_val_acc_train_loss, best_val_acc_train_acc = best_model.evaluate(X_train, y_train)
            best_val_acc_train_losses.append(best_val_acc_train_loss)
            best_val_acc_train_accs.append(best_val_acc_train_acc)

            # store results
            search_results.append({
                **params,
                'best_val_acc': np.mean(best_val_accs),
                'best_val_acc_sem': np.std(best_val_accs)/np.sqrt(k),
                'best_val_acc_train_acc': np.mean(best_val_acc_train_accs),
                'best_val_acc_train_acc_sem': np.std(best_val_acc_train_accs)/np.sqrt(k),
                'best_val_acc_loss': np.mean(best_val_acc_losses),
                'best_val_acc_loss_sem': np.std(best_val_acc_losses)/np.sqrt(k),
                'best_val_acc_train_loss': np.mean(best_val_acc_train_losses),
                'best_val_acc_train_loss_sem': np.std(best_val_acc_train_losses)/np.sqrt(k),
                'history': fit_results.history
            })

    resultsDF = pd.DataFrame(search_results)
    resultsDF = resultsDF.sort_values('best_val_acc', ascending=False)
    if i == 0:
        resultsDF.to_csv('results_f.csv')
    else:
        resultsDF.to_csv('results_m.csv')

    #plot results
    top_3_indices = resultsDF.index.values[:3]
    plt.plot([],[],'k--', label='Training')
    plt.plot([],[],'k-', label='Validation')

    for idx, (row_index, row_series) in enumerate(resultsDF.head(3).iterrows()):
        x = np.arange(1, len(row_series['history']['loss'])+1)
        parameter_combination_string = f"$n_\\mathrm{{layers}}=${row_series['num_layers']}"
        plt.plot(x, row_series['history']['loss'], '--', color=f'C{idx}')
        plt.plot(x, row_series['history']['val_loss'], '-', color=f'C{idx}')

        plt.fill_between([],[],[],color=f'C{idx}', label=parameter_combination_string)

    plt.xlabel('Epochs')
    plt.ylabel('Categorical crossentropy loss')
    plt.legend()
    if i == 0:
        plt.savefig('plots/loss_f.pdf')
    else:
        plt.savefig('plots/loss_m.pdf')
    
    plt.close()

    top_3_indices = resultsDF.index.values[:3]
    plt.plot([],[],'k--', label='Training')
    plt.plot([],[],'k-', label='Validation')
    for idx, (row_index, row_series) in enumerate(resultsDF.head(3).iterrows()):
        x = np.arange(1, len(row_series['history']['accuracy'])+1)
        parameter_combination_string = f"$n_\\mathrm{{layers}}=${row_series['num_layers']}"
        plt.plot(x, row_series['history']['accuracy'], '--', color=f'C{idx}')
        plt.plot(x, row_series['history']['val_accuracy'], '-', color=f'C{idx}')

        plt.fill_between([],[],[],color=f'C{idx}', label=parameter_combination_string)


    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    if i == 0:
        plt.savefig('plots/acc_f.pdf')
    else:
        plt.savefig('plots/acc_m.pdf')
    plt.close()
    
    i+=1