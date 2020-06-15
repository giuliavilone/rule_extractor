from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
from scipy.stats import mode
from sklearn.metrics import accuracy_score
import sys

# Global variables
n_members = 10

#from sklearn.datasets import load_iris
#from sklearn.tree import DecisionTreeClassifier
#from sklearn.tree.export import export_text
#iris = load_iris()
#decision_tree = DecisionTreeClassifier(random_state=0, max_depth=2)
#decision_tree = decision_tree.fit(iris.data, iris.target)
#r = export_text(decision_tree, feature_names=iris['feature_names'])
#print(r)

# Functions
def load_all_models(n_models):
    """
    This function returns the list of the trained models
    :param n_models:
    :return: list of trained models
    """
    all_models = list()
    for i in range(n_models):
        # define filename for this ensemble
        filename = 'c45_model_' + str(i + 1) + '.h5'
        # load model from file
        model = load_model(filename)
        # add to list of members
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # combining the members via plurality voting
    voted_yhats = np.argmax(yhats, axis=2)
    results = mode(voted_yhats, axis=0)[0]
    return results

def synthetic_data_generator(indf, n_samples):
    """
    Given an input dataframe, the function returns a new dataframe containing random numbers
    generated within the value ranges of the input attributes.
    :param indf:
    :param n_samples: integer number of samples to be generated
    :return: outdf: of synthetic data
    """
    outdf = pd.DataFrame()
    for column in indf.columns.tolist():
        minvalue = indf.min()[column]
        maxvalue = indf.max()[column]
        outdf[column] = np.round(np.random.uniform(minvalue,maxvalue,n_samples),1)
    return outdf


# Main code
data = arff.loadarff('datasets-UCI/UCI/iris.arff')
data = pd.DataFrame(data[0])


# Separating independent variables from the target one
X = data.drop(columns=['class']).to_numpy()
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())

# define model
model = Sequential()
model.add(Dense(20, input_dim=X.shape[-1], activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

fold_var = 1
for _ in range(n_members):
    # select indexes
    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=True, n_samples=int(len(X)*0.7))
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    checkpointer = ModelCheckpoint(filepath='c45_model_'+str(fold_var)+'.h5',
                                  save_weights_only=False,
                                  monitor='loss',
                                  save_best_only=True,
                                  verbose=1)
    history = model.fit(X_train, to_categorical(y_train, num_classes=3),
                        validation_data=(X_test, to_categorical(y_test, num_classes=3)),
                        epochs=50,
                        callbacks=[checkpointer])

    fold_var += 1

# load all models
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

ensemble_res = ensemble_predictions(members, X)
print(accuracy_score(ensemble_res[0], y))

# Same process of REFNE
synth_samples = X.shape[0] * 2
xSynth = synthetic_data_generator(X, synth_samples)
ySynth = ensemble_predictions(members, xSynth)