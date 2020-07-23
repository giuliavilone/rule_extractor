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
import sys
from collections import Counter
import random

# Global variables
n_members = 5

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


def chimerge(data, attr, label):
    """
    Code copied from https://gist.github.com/alanzchen/17d0c4a45d59b79052b1cd07f531689e
    :param data:
    :param attr:
    :param label:
    :param max_intervals:
    :return:
    """
    distinct_vals = sorted(set(data[attr])) # Sort the distinct values
    labels = sorted(set(data[label])) # Get all possible labels
    empty_count = {l: 0 for l in labels} # A helper function for padding the Counter()
    # Initialize the intervals for each attribute
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
    more_merges = True
    while more_merges: # While loop
        chi = []
        for i in range(len(intervals)-1):
            lab0 = sorted(set(data[label][data[attr].between(intervals[i][0], intervals[i][1])]))
            lab1 = sorted(set(data[label][data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]))
            if len(lab0) + len(lab1) > 2 or lab0 != lab1:
            # if lab0 != lab1:
                continue
            else:
                # Calculate the Chi2 value
                obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
                obs1 = data[data[attr].between(intervals[i+1][0], intervals[i+1][1])]
                total = len(obs0) + len(obs1)
                count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
                count_total = count_0 + count_1
                expected_0 = count_total*sum(count_0)/total
                expected_1 = count_total*sum(count_1)/total
                chi_ = (count_0 - expected_0)**2/expected_0 + (count_1 - expected_1)**2/expected_1
                chi_ = np.nan_to_num(chi_) # Deal with the zero counts
                chi.append(sum(chi_)) # Finally do the summation for Chi2
        if len(chi) == 0:
            more_merges = False
            break
        else:
            min_chi = min(chi) # Find the minimal Chi2 for current iteration
            min_chi_index = -1
            for i, v in enumerate(chi):
                if v == min_chi:
                    min_chi_index = i # Find the index of the interval to be merged
                    break
            new_intervals = [] # Prepare for the merged new data array
            skip = False
            done = False
            for j in range(len(intervals)):
                if skip:
                    skip = False
                    continue
                if j == min_chi_index and not done: # Merge the intervals
                    t = intervals[j] + intervals[j+1]
                    new_intervals.append([min(t), max(t)])
                    skip = True
                    done = True
                else:
                    new_intervals.append(intervals[j])
            intervals = new_intervals
    return intervals

def discretizer(indf, intervals):
    """
    The function takes a continuous attribute of a dataframe and the list of the intervals, then it performs the
    discretization of the attribute
    :param indf:
    :param intervals:
    :return: indf: with discrete values
    """
    indf = np.array(indf)
    for i in range(len(intervals)):
        minVal = intervals[i][0]
        maxVal = intervals[i][1]
        indf[(indf >= minVal) & (indf <= maxVal)] = minVal
    return indf.tolist()


def rule_maker(df, intervals):
    """
    Creates the IF-THEN rules
    :param df:
    :return: rules
    """
    # Getting list of columns of input dataframe and randomly shuffling them
    col = df.columns.values.tolist()
    col.remove('class')
    random.shuffle(col)
    for item in col:
        interv = intervals[item]
        attr_list = df[[item, 'class']].groupby(item)['class'].nunique().reset_index(drop=False)
        print(interv)
        print(attr_list)


# Main code
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

    fold_var += 1

# load all models
members = load_all_models(n_members)
print('Loaded %d models' % len(members))

ensemble_res = ensemble_predictions(members, X)
print(accuracy_score(ensemble_res[0], y))

# According to the paper, it is enough to generate a new dataset which is twice the training set to obtain
# very accurate rules
synth_samples = X.shape[0] * 2
xSynth = synthetic_data_generator(X, synth_samples)
ySynth = ensemble_predictions(members, xSynth)

# Discretizing the continuous attributes
attr_list = xSynth.columns.tolist()
totSynth = xSynth
totSynth['class'] = ySynth[0]
interv_dict = {}
for attr in attr_list:
    interv = chimerge(data=totSynth, attr=attr, label='class')
    totSynth[attr] = discretizer(totSynth[attr], interv)
    interv_dict[attr] = interv

rule_maker(totSynth, interv_dict)