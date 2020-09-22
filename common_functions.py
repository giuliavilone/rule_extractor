from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def create_model_old(train_x, num_classes, hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=train_x.shape[1], activation="sigmoid"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


def create_model(train_x, n_classes, neurons, optimizer='Adam', init_mode='uniform',
                 activation='softmax', dropout_rate=0.0, weight_constraint=0
                 ):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_x.shape[1], activation=activation, kernel_initializer=init_mode,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


def model_train(train_x, train_y, test_x, test_y, model, model_name, n_epochs=100, batch_size=10):
    check_pointer = ModelCheckpoint(filepath=model_name,
                                   save_weights_only=False,
                                   monitor='accuracy',
                                   save_best_only=True,
                                   verbose=1)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size,
              epochs=n_epochs, callbacks=[check_pointer]
              )
    return model, history


def perturbator(indf, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :type indf: Pandas dataframe
    """
    noise = np.random.normal(mu, sigma, indf.shape)
    return indf + noise


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # combining the members via plurality voting
    voted_yhats = np.argmax(yhats, axis=2)
    results = mode(voted_yhats, axis=0)[0]
    return results


def dataset_uploader(item, target_var='class'):
    le = LabelEncoder()
    dataset = pd.read_csv('datasets-UCI/Used_data/' + item['dataset'] + '.csv')
    col_types = dataset.dtypes
    out_disc = []
    out_cont = []
    for index, value in col_types.items():
        if value == 'object':
            dataset[index] = le.fit_transform(dataset[index].tolist())
            if index != 'class':
                out_disc.append(index)
        else:
            if index != 'class':
                out_cont.append(index)
    # Separating independent variables from the target one
    # X = dataset.drop(columns=[target_var]).to_numpy()
    X = dataset.drop(columns=[target_var])
    y = le.fit_transform(dataset[target_var].tolist())
    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=False, n_samples=int(len(X) * 0.7))
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
    # X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    return X_train, X_test, y_train, y_test, out_disc, out_cont
