from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.regularizers import l2


# Function to create model, required for KerasClassifier
def create_model(n_layers, units, activation, l_2, learning_rate):
    # create model
    model = Sequential()
    model.add(keras.layers.Flatten(input_shape=(21,)))
    for i in range(n_layers):
        if l2:
            model.add(keras.layers.Dense(units=eval("units"),
                                         activation=eval("activation"),
                                         kernel_regularizer=l2(0.0001)
                                         ))
        else:
            model.add(keras.layers.Dense(units=eval("units"),
                                         activation=eval("activation"),
                                         # kernel_regularizer=l2(0.0001)
                                         ))
        # model.add(keras.layers.Dropout(eval("Dropout")))

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
