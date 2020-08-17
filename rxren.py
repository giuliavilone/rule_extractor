from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
from keras.models import load_model
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, create_model, model_trainer

# Global variable
TOLERANCE = 0.01
MODEL_NAME = 'rxren_model.h5'

data = arff.loadarff('datasets-UCI/UCI/diabetes.arff')
data = pd.DataFrame(data[0])
n_classes = 2
hidden_neurons = 3

# Functions
def input_delete(insignificant_index, inDf, inWeight=None):
    """
    Delete the variable of the input vector corresponding the insignificant input neurons and, if required, the
    corresponding weights of the neural network
    :param inDf:
    :param inDf:
    :param inWeight:
    :return: the trimmed weights and input vector
    """
    outDf = copy.deepcopy(inDf)
    outDf = np.delete(outDf, insignificant_index, 1)
    outWeight = None
    if inWeight is not None:
        outWeight = copy.deepcopy(inWeight)
        outWeight[0] = np.delete(outWeight[0], insignificant_index, 0)
    return outDf, outWeight

def model_pruned_prediction(inputX, w):
    new_m = create_model(inputX, n_classes, hidden_neurons)
    new_m.set_weights(w)
    results = new_m.predict(inputX)
    results = np.argmax(results, axis=1)
    return results

def network_pruning(w, cX, cy, train_x, train_y, accuracy):
    """
    This function removes the insignificant input neurons
    :param w: model's weights
    :param t: theta
    :param cX: correctX
    :param cy: correcty
    :return: pruned weights
    """
    insignificant_neurons = []
    missclassified_list = []
    error_list = []
    new_acc = 0
    for i in range(w[0].shape[0]):
        newCx, new_w = input_delete(i, cX, inWeight=w)
        res = model_pruned_prediction(newCx, new_w)
        misclassified = [i for i in range(len(cy)) if res[i] != cy[i]]
        missclassified_list.append(misclassified)
        err = len(misclassified)
        error_list.append(err)
    pruning = True
    theta = min(error_list)
    while pruning:
        insignificant_neurons = insignificant_neurons + [i for i, e in enumerate(error_list) if e == theta]
        new_train_x, new_w = input_delete(insignificant_neurons, train_x, inWeight=w)
        new_res = model_pruned_prediction(new_train_x, new_w)
        new_acc = accuracy_score(new_res, train_y)
        if new_acc >= accuracy - TOLERANCE:
            new_error_list = [e for i, e in enumerate(error_list) if e > theta]
            # Leaving at least one significant neuron
            if len(new_error_list) > 1:
                theta = min(new_error_list)
            else:
                pruning = False
        else:
            pruning = False
    return missclassified_list, sorted(insignificant_neurons), new_acc, theta


def rule_limits_calculator(c_x, c_y, missclassified_list, significant_neurons, error, alpha=0.5):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[missclassified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'neuron': i, 'class': k,
                               'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                max(miss_class[:, i][miss_class[:, -1] == k])]}
                                for k in np.unique(miss_class[:, -1])
                                if len(miss_class[:, i][miss_class[:, -1] == k]) > (error * alpha)]
    return grouped_miss_class

def rule_evaluator(train_x, train_y, rule_list, class_list):
    predicted_y = np.empty((train_y.shape))
    predicted_y[:] = np.NaN
    for rule in rule_list:
        if rule['class'] in class_list[:]:
            class_list = np.delete(class_list, rule['class'])
        col = rule['neuron']
        minimum = rule['limits'][0]
        maximum = rule['limits'][1]
        predicted_y[(train_x[:, col] >= minimum) * (train_x[:, col] <= maximum)] = int(rule['class'])
    if len(class_list) == 1:
        predicted_y[np.isnan(predicted_y)] = class_list[0]
    else:
        print("It is not possible to identify default class")
    rule_accuracy = accuracy_score(predicted_y, train_y)
    # Calculate min and max of mismatched instances
    mismatched = [index for index, elem in enumerate(train_y) if elem != predicted_y[index]]
    ret_rules = []
    for rule in rule_list:
        col = rule['neuron']
        mismatched_values = train_x[mismatched, col]
        rule['new_limits'] = [min(mismatched_values), max(mismatched_values)]
        ret_rules.append(rule)
    return rule_accuracy, ret_rules


# Main code

# Separating independent variables from the target one
X = data.drop(columns=['class']).to_numpy()
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())


ix = [i for i in range(len(X))]
train_index = resample(ix, replace=True, n_samples=int(len(X)*0.7), random_state=0)
val_index = [x for x in ix if x not in train_index]
X_train, X_test = X[train_index], X[val_index]
y_train, y_test = y[train_index], y[val_index]

# define model
model = create_model(X, n_classes, hidden_neurons)

model_train = True
if model_train:
    model_trainer(X_train, to_categorical(y_train, num_classes=n_classes),
                  X_test, to_categorical(y_test, num_classes=n_classes), model, MODEL_NAME)

model = load_model(MODEL_NAME)
weights = np.array(model.get_weights())
results = model.predict(X)
results = np.argmax(results, axis=1)
correctX = X[[results[i] == y[i] for i in range(len(y))]]
correcty = y[[results[i] == y[i] for i in range(len(y))]]

new_res = model.predict(X_train)
new_res = np.argmax(new_res, axis=1)
acc = accuracy_score(new_res, y_train)
print(acc)

miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_train, y_train, acc)
significant_index = [i for i in range(weights[0].shape[0]) if i not in ins_index]
print(new_accuracy)
rule_limits = rule_limits_calculator(correctX, correcty, miss_list, significant_index, err, alpha=0.5)
# print(rule_limits)

rule_simplifier = True
old_rule_acc = new_accuracy
new_limits = []
while rule_simplifier:
    rule_acc, new_limits = rule_evaluator(X_train, y_train, rule_limits, np.unique(y))
    if rule_acc > old_rule_acc:
        old_rule_acc = rule_acc
    else:
        rule_simplifier = False
    new_limits[0]['limits'] = new_limits[0].pop('new_limits')

print(new_limits)
predicted_labels = np.argmax(model.predict(X_test), axis=1)

num_test_examples = X_test.shape[0]
perturbed_data = perturbator(X_test)
rule_labels = np.empty(num_test_examples)
rule_labels[:] = np.nan
perturbed_labels = np.empty(num_test_examples)
perturbed_labels[:] = np.nan
for rule in new_limits:
    neuron = X_test[:, rule['neuron']]
    indexes = np.where((neuron >= rule['limits'][0]) & (neuron <= rule['limits'][1]))
    rule_labels[indexes] = rule['class']
    p_neuron = perturbed_data[:, rule['neuron']]
    indexes = np.where((p_neuron >= rule['limits'][0]) & (p_neuron <= rule['limits'][1]))
    perturbed_labels[indexes] = rule['class']

rule_labels[np.where(np.isnan(rule_labels))] = 1
perturbed_labels[np.where(np.isnan(perturbed_labels))] = 1

correct = 0
fidel = 0
rob = 0
for i in range(0, num_test_examples):
    fidel += (rule_labels[i] == predicted_labels[i])
    correct += (rule_labels[i] == y_test[i])
    rob += (predicted_labels[i] == perturbed_labels[i])

fidelity = fidel / num_test_examples
print("Fidelity of the ruleset is : " + str(fidelity))
completeness = len(rule_labels) / num_test_examples
print("Completeness of the ruleset is : " + str(completeness))
correctness = correct / num_test_examples
print("Correctness of the ruleset is : " + str(correctness))
robustness = rob / num_test_examples
print("Robustness of the ruleset is : " + str(robustness))