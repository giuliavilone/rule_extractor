from scipy.io import arff
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score

# Global variables
n_members = 5

def load_all_models(n_models):
    """
    This function returns the list of the trained models
    :param n_models:
    :return: list of trained models
    """
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # combining the members via plurality voting
    voted_yhats = np.argmax(yhats, axis=2)
    results = mode(voted_yhats, axis=0)[0]
    return results

data = arff.loadarff('datasets-UCI/UCI/iris.arff')
data = pd.DataFrame(data[0])

# Separating independent variables from the target one
X = data.drop(columns=['class'])
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())

# Create the object to perform cross validation
skf = StratifiedKFold(n_splits=n_members, random_state=7, shuffle=True)

# define model
model = Sequential()
model.add(Dense(20, input_dim=X.shape[-1], activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model on the 5 cross validation datasets
fold_var = 1
VAL_ACCURACY = []
for train_index, val_index in skf.split(X,y):
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
    y_train, y_test = y[train_index], y[val_index]
    checkpointer = ModelCheckpoint(filepath='model_'+str(fold_var)+'.h5',
                                  save_weights_only=False,
                                  monitor='loss',
                                  save_best_only=True,
                                  verbose=1)
    history = model.fit(X_train, to_categorical(y_train, num_classes=3),
                        validation_data=(X_test, to_categorical(y_test, num_classes=3)),
                        epochs=50,
                        callbacks=[checkpointer])
    #Evaluate the performance of the model
    y_test_enc = to_categorical(y_test)
    _, acc = model.evaluate(X_test, y_test_enc, verbose=0)
    print('Model Accuracy: %.3f' % acc)
    VAL_ACCURACY.append(acc)
    fold_var += 1

# load all models
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

ensemble_res = ensemble_predictions(members, X)
print(accuracy_score(ensemble_res[0], y))

