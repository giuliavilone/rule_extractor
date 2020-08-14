from scipy.io import arff
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np
import sys
from collections import Counter
import random
import copy

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
        outdf[column] = np.round(np.random.uniform(minvalue, maxvalue, n_samples), 1)
    return outdf


def chimerge(data, attr, label):
    """
    Code copied from https://gist.github.com/alanzchen/17d0c4a45d59b79052b1cd07f531689e
    :param data:
    :param attr:
    :param label:
    :return:
    """
    distinct_vals = sorted(set(data[attr]))  # Sort the distinct values
    labels = sorted(set(data[label]))  # Get all possible labels
    empty_count = {l: 0 for l in labels}  # A helper function for padding the Counter()
    # Initialize the intervals for each attribute
    intervals = [[distinct_vals[i], distinct_vals[i]] for i in range(len(distinct_vals))]
    more_merges = True
    while more_merges:  # While loop
        chi = []
        for i in range(len(intervals) - 1):
            lab0 = sorted(set(data[label][data[attr].between(intervals[i][0], intervals[i][1])]))
            lab1 = sorted(set(data[label][data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]))
            # if len(lab0) + len(lab1) > 2 or lab0 != lab1:
            if lab0 != lab1:
                chi.append(1000.0)
                continue
            else:
                # Calculate the Chi2 value
                obs0 = data[data[attr].between(intervals[i][0], intervals[i][1])]
                obs1 = data[data[attr].between(intervals[i + 1][0], intervals[i + 1][1])]
                total = len(obs0) + len(obs1)
                count_0 = np.array([v for i, v in {**empty_count, **Counter(obs0[label])}.items()])
                count_1 = np.array([v for i, v in {**empty_count, **Counter(obs1[label])}.items()])
                count_total = count_0 + count_1
                expected_0 = count_total * sum(count_0) / total
                expected_1 = count_total * sum(count_1) / total
                chi_ = (count_0 - expected_0) ** 2 / expected_0 + (count_1 - expected_1) ** 2 / expected_1
                chi_ = np.nan_to_num(chi_)  # Deal with the zero counts
                chi.append(sum(chi_))  # Finally do the summation for Chi2

        min_chi = min(chi)  # Find the minimal Chi2 for current iteration
        if min_chi == 1000.0:
            break
        min_chi_index = -1
        for i, v in enumerate(chi):
            if v == min_chi:
                min_chi_index = i  # Find the index of the interval to be merged
                break
        new_intervals = []  # Prepare for the merged new data array
        skip = False
        done = False
        for j in range(len(intervals)):
            if skip:
                skip = False
                continue
            if j == min_chi_index and not done:  # Merge the intervals
                t = intervals[j] + intervals[j + 1]
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


def select_random_item(int_list, ex_item_list):
    ex_list = copy.deepcopy(int_list)
    for i in ex_item_list:
        ex_list.remove(i)
    new_item = random.choice(ex_list)
    return new_item


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
    ret = []
    for item in col:
        attr_list = df[[item, 'class']].groupby(item).agg(unique_class=('class', 'nunique'),
                                                          max_class=('class', 'max')
                                                          ).reset_index(drop=False)
        new_rules = attr_list[attr_list['unique_class'] == 1]
        if len(new_rules) == 0:
            item1 = select_random_item(col, [item])
            attr_list = df[[item, item1, 'class']].groupby([item, item1]).agg(unique_class=('class', 'nunique'),
                                                                              max_class=('class', 'max')
                                                                              ).reset_index(drop=False)
            new_rules = attr_list[attr_list['unique_class'] == 1]
            if len(new_rules) == 0:
                item2 = select_random_item(col, [item, item1])
                attr_list = df[[item, item1, item2, 'class']].groupby([item, item1, item2]).agg(
                    unique_class=('class', 'nunique'),
                    max_class=('class', 'max')
                ).reset_index(drop=False)
                new_rules = attr_list[attr_list['unique_class'] == 1]
                if len(new_rules) == 0:
                    continue
        new_col = new_rules.columns.values.tolist()
        new_col.remove('unique_class')
        new_col.remove('max_class')
        for index, row in new_rules.iterrows():
            new_dict = {'neuron':new_col}
            new_dict['class'] = row['max_class'].tolist()
            rule_intervals = []
            for c in new_col:
                interv = intervals[c]
                rule_int = [item for item in interv if item[0] == row[c]]
                rule_intervals.append(rule_int)
            new_dict['limits'] = rule_intervals
        ret.append(new_dict)
    return ret

def perturbator(indf, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :type indf: Pandas dataframe
    """
    noise = np.random.normal(mu, sigma, indf.shape)
    return indf + noise

def rule_applier(indf, iny, rules):
    """
    Apply the input rules to the list of labels iny according to the rule conditions on indf
    :param indf:
    :param iny:
    :param rules:
    :return: iny
    """
    indexes = []
    for r in range(len(rules['neuron'])):
        x = indf[rules['neuron'][r]]
        ix = np.where((x >= rules['limits'][r][0][0]) & (x <= rules['limits'][r][0][1]))[0]
        indexes = [x for x in ix if x not in indexes]

    iny[indexes] = rule['class']
    return iny

# Main code
data, meta = arff.loadarff('datasets-UCI/UCI/iris.arff')
data = pd.DataFrame(data)
le = LabelEncoder()
# Encoding the nominal fields
for item in range(len(meta.names())):
    item_name = meta.names()[item]
    item_type = meta.types()[item]
    if item_type == 'nominal':
        data[item_name] = le.fit_transform(data[item_name].tolist())

# Separating independent variables from the target one
X = data.drop(columns=['class'])
y = data['class']

# Extracting the classes to be predicted
classes = y.unique()

# Create the object to perform cross validation
skf = StratifiedKFold(n_splits=n_members, random_state=7, shuffle=True)

# define model
model = Sequential()
model.add(Dense(20, input_dim=X.shape[-1], activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Training the model on the 5 cross validation datasets
fold_var = 1
model_train = False
if model_train:
    for train_index, val_index in skf.split(X, y):
        X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
        y_train, y_test = y[train_index], y[val_index]
        checkpointer = ModelCheckpoint(filepath='model_' + str(fold_var) + '.h5',
                                       save_weights_only=False,
                                       monitor='loss',
                                       save_best_only=True,
                                       verbose=1)
        history = model.fit(X_train, to_categorical(y_train, num_classes=3),
                            validation_data=(X_test, to_categorical(y_test, num_classes=3)),
                            epochs=50,
                            callbacks=[checkpointer])

        fold_var += 1
else:
    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=True, n_samples=int(len(X) * 0.7), random_state=0)
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
    y_train, y_test = y[train_index], y[val_index]

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

final_rules = rule_maker(totSynth, interv_dict)
print(final_rules)

predicted_labels = np.argmax(model.predict(X_test), axis=1)

num_test_examples = X_test.shape[0]
perturbed_data = perturbator(X_test)
rule_labels = np.empty(num_test_examples)
rule_labels[:] = np.nan
perturbed_labels = np.empty(num_test_examples)
perturbed_labels[:] = np.nan

for rule in final_rules:
    neuron = X_test[rule['neuron']]
    rule_labels = rule_applier(neuron, rule_labels, rule)
    p_neuron = perturbed_data[rule['neuron']]
    perturbed_labels = rule_applier(p_neuron, rule_labels, rule)

completeness = sum(~np.isnan(rule_labels)) / num_test_examples
print("Completeness of the ruleset is : " + str(completeness))

rule_labels[np.where(np.isnan(rule_labels))] = max(classes) + 10
perturbed_labels[np.where(np.isnan(perturbed_labels))] = max(classes) + 10

correct = 0
fidel = 0
rob = 0
y_test = y_test.tolist()
for i in range(0, num_test_examples):
    fidel += (rule_labels[i] == predicted_labels[i])
    correct += (rule_labels[i] == y_test[i])
    rob += (predicted_labels[i] == perturbed_labels[i])

fidelity = fidel / num_test_examples
print("Fidelity of the ruleset is : " + str(fidelity))
correctness = correct / num_test_examples
print("Correctness of the ruleset is : " + str(correctness))
robustness = rob / num_test_examples
print("Robustness of the ruleset is : " + str(robustness))