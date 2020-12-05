import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, model_train, dataset_uploader, create_model, rule_metrics_calculator
import dictlib
from sklearn.model_selection import train_test_split
import sys

# np.random.seed(3)
np.random.seed(1)


# Functions
def prediction_reshape(prediction_list):
    if len(prediction_list.shape) > 1:
        ret = np.argmax(prediction_list, axis=1)
    else:
        ret = np.reshape(prediction_list, -1).tolist()
        ret = [round(x) for x in ret]
    return ret


def input_delete(insignificant_index, in_df, in_weight=None):
    """
    Delete the variable of the input vector corresponding the insignificant input neurons and, if required, the
    corresponding weights of the neural network
    :param insignificant_index:
    :param in_df:
    :param in_weight:
    :return: the trimmed weights and input vector
    """
    out_df = copy.deepcopy(in_df)
    out_df = np.delete(out_df, insignificant_index, 1)
    out_weight = None
    if in_weight is not None:
        out_weight = copy.deepcopy(in_weight)
        out_weight[0] = np.delete(out_weight[0], insignificant_index, 0)
    return out_df, out_weight


def prediction_classifier(orig_y, predict_y, comparison="misclassified"):
    ret = {}
    out_classes = np.unique(orig_y)
    for cls in out_classes:
        if comparison == 'misclassified':
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] != orig_y[i] and orig_y[i] == cls]
        else:
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] == orig_y[i] and orig_y[i] == cls]
    return ret


def model_pruned_prediction(insignificant_index, in_df, in_item, in_weight=None):
    """
    Calculate the output classes predicted by the pruned model.
    :param insignificant_index: list of the indexes of insignificant input neurons
    :param in_df: input dataframe including independent variables
    :param in_item: hype-parameters of the model
    :param in_weight: weight of trained model
    :return: numpy array with the output classes predicted by the pruned model
    """
    input_x, w = input_delete(insignificant_index, in_df, in_weight=in_weight)
    new_m = create_model(input_x, in_item['classes'], in_item['neurons'], eval(in_item['optimizer']),
                         in_item['init_mode'], in_item['activation'], in_item['dropout_rate'],
                         eval(in_item['weight_constraint']), loss='binary_crossentropy',
                         out_activation='sigmoid')
    new_m.set_weights(w)
    ret = new_m.predict(input_x)
    ret = prediction_reshape(ret)
    return ret


def network_pruning(w, correct_x, correct_y, test_x, test_y, accuracy, columns, in_item=None):
    """
    Remove the insignificant input neurons of the input model, based on the weight w.
    :param w: model's weights
    :param correct_x: set of correctly classified instances (independent variables)
    :param correct_y: set of correctly classified instances (dependent variable)
    :param test_x: test dataset (independent variables)
    :param test_y: test dataset (dependent variable)
    :param accuracy: accuracy
    :param in_item: in_item
    :return: miss-classified instances, pruned weights, accuracy of pruned model,
    """
    temp_w = copy.deepcopy(w)
    temp_x = copy.deepcopy(correct_x)
    temp_test_x = copy.deepcopy(test_x)
    significant_cols = copy.deepcopy(columns)
    new_acc = accuracy
    miss_classified_dict = {}
    theta = 0
    pruning = True
    while pruning:
        miss_classified_dict = {}
        error_list = []
        for i in range(temp_w[0].shape[0]):
            res = model_pruned_prediction(i, temp_x, in_item, in_weight=temp_w)
            misclassified = prediction_classifier(correct_y, res)
            miss_classified_dict[i] = misclassified
            error_list.append(sum([len(value) for key, value in misclassified.items()]))
        # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
        # and its accuracy must be set equal to the accuracy of the original network
        if sum(error_list) > 0:
            theta = min(error_list)
            insignificant_neurons_temp = [i for i, e in enumerate(error_list) if e == theta]
            new_res = model_pruned_prediction(insignificant_neurons_temp, temp_test_x, in_item, in_weight=temp_w)
            new_acc_temp = accuracy_score(new_res, test_y)
            if new_acc_temp >= accuracy:
                significant_cols = list(np.delete(np.array(significant_cols), insignificant_neurons_temp))
                new_acc = new_acc_temp
                temp_x, temp_w = input_delete(insignificant_neurons_temp, temp_x, in_weight=temp_w)
                temp_test_x, _ = input_delete(insignificant_neurons_temp, temp_test_x)
            else:
                pruning = False
        else:
            pruning = False

    return miss_classified_dict, temp_x, temp_w, new_acc, theta, significant_cols


def correct_examples_finder(correct_x, correct_y, in_item, in_weight=None):
    """
    This function finds the examples correctly classified by the pruned network for each significant input neuron. The
    paper is not clear on how to do that, so my interpretation is that these are the examples correctly classified
    when the significant neuron under analysis is removed, as done for the misclassified examples.
    :param correct_x: correctX
    :param correct_y: correctX
    :param in_item: correctX
    :param in_weight: correctX
    :return: dictionary with the indexes of the neurons as keys and the list of the examples correctly classified for
    each of them
    """
    ret = {}
    for item in range(correct_x.shape[1]):
        res = model_pruned_prediction([item], correct_x, in_item, in_weight=in_weight)
        ret[item] = prediction_classifier(correct_y, res, comparison="correct")
    return ret


def combine_dict_list(dict_1, dict_2):
    ret = {}
    for key, value in dict_1.items():
        temp_ret = dictlib.union_setadd(dict_1[key], dict_2[key])
        for key2, value2 in temp_ret.items():
            temp_ret[key2] = list(set(value2))
        ret[key] = temp_ret
    return ret


def rule_limits_calculator(c_x, c_y, classified_dict, alpha=0.5):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = {k: [] for k in np.unique(c_y)}
    for i in range(c_x.shape[1]):
        mp = sum([len(value) for key, value in classified_dict[i].items()])
        for k in np.unique(c_y):
            ucm_class = c_tot[classified_dict[i][k]]
            if len(ucm_class[:, i]) > (mp * alpha):
                # Splitting the misclassified input values according to their output classes
                grouped_miss_class[k] += [{'neuron': i, 'limits': [min(ucm_class[:, i]), max(ucm_class[:, i])]}]
    return grouped_miss_class


def rule_elicitation(x, pred_y, rule_list, cls):
    for item in rule_list:
        minimum = item['limits'][0]
        maximum = item['limits'][1]
        pred_y[(x[:, item['neuron']] >= minimum) * (x[:, item['neuron']] <= maximum)] = cls
    return pred_y


def ruleset_accuracy(x_arr, y_list, rule_set, cls, classes):
    predicted_y = np.empty(x_arr.shape[0])
    predicted_y[:] = np.NaN
    predicted_y = rule_elicitation(x_arr, predicted_y, rule_set, cls)
    predicted_y[np.isnan(predicted_y)] = classes + 10
    ret = accuracy_score(y_list, predicted_y)
    return ret


def rule_pruning(train_x, train_y, rule_set, classes_n):
    """
    Pruning the rules
    :param train_x:
    :param train_y:
    :param rule_set:
    :param classes_n:
    :return:
    """
    ret = {}
    orig_acc = {}
    for cls, rule_list in rule_set.items():
        orig_acc[cls] = ruleset_accuracy(train_x, train_y, rule_list, cls, classes_n)
        ix = 0
        while len(rule_list) > 1 and ix < len(rule_list):
            new_rule = [j for i, j in enumerate(rule_list) if i != ix]
            new_acc = ruleset_accuracy(train_x, train_y, new_rule, cls, classes_n)
            if new_acc >= orig_acc[cls]:
                rule_list.pop(ix)
                orig_acc[cls] = new_acc
            else:
                ix += 1
        ret[cls] = rule_list
    return ret, orig_acc


def rule_evaluator(x, y, rule_dict, orig_acc, class_list):
    ret = copy.deepcopy(rule_dict)
    predicted_y = np.empty(x.shape[0])
    predicted_y[:] = np.NaN
    for cls, rule_list in rule_dict.items():
        predicted_y = rule_elicitation(x, predicted_y, rule_list, cls)
    predicted_y[np.isnan(predicted_y)] = len(class_list) + 10
    for cls, rule_list in rule_dict.items():
        print('Working on class: ', cls)
        ixs = np.where(predicted_y == cls)[0].tolist()
        if len(ixs) > 0:
            for pos in range(len(rule_list)):
                item = rule_list[pos]
                new_min = min(x[ixs, item['neuron']])
                new_max = max(x[ixs, item['neuron']])
                ret[cls][pos] = {'neuron': item['neuron'], 'limits': [new_min, new_max]}
                new_acc = ruleset_accuracy(x, y, ret[cls], cls, len(np.unique(y)))
                if new_acc < orig_acc[cls]:
                    ret[cls][pos] = rule_dict[cls][pos]
    return ret


def rule_size_calculator(rule_set):
    avg = 0
    length = 0
    for key, value in rule_set.items():
        length += len(value)
        avg += 1
    avg = avg / length
    return avg, length

# Main code

# Loading dataset
# Alpha is set equal to the percentage of input instances belonging to the least-represented class in the dataset
alpha = 0.4
parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
dataset_par = parameters.iloc[10]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')

MODEL_NAME = 'trained_model_' + dataset_par['dataset'] + '.h5'
X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33)

column_lst = X_train.columns.tolist()
column_dict = {i: column_lst[i] for i in range(len(column_lst))}

y = np.concatenate((y_train, y_test), axis=0)
X_train, X_test, X_val = X_train.to_numpy(), X_test.to_numpy(), X_val.to_numpy()
n_classes = dataset_par['classes']

model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                     dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                     weight_constraint=eval(dataset_par['weight_constraint'])
                     )
model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
            X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model, MODEL_NAME,
            n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size']
            )

# model = load_model(MODEL_NAME)
weights = np.array(model.get_weights())
results = prediction_reshape(model.predict_classes(X_train))

# This will be used for calculating the final metrics
predicted_labels = prediction_reshape(model.predict(X_test))

correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
print('Number of correctly classified examples', correctX.shape)
correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
acc = accuracy_score(results, y_train)
print("Accuracy of original model on the train dataset: ", acc)
test_pred = prediction_reshape(model.predict(X_val))
test_acc = accuracy_score(test_pred, y_val)
print("Accuracy of original model on the validation dataset: ", test_acc)

miss_dict, pruned_x, pruned_w, new_accuracy, err, sig_cols = network_pruning(weights, correctX, correcty, X_val,
                                                                             y_val, test_acc, column_lst,
                                                                             in_item=dataset_par)

print("Accuracy of pruned network", new_accuracy)
corr_dict = correct_examples_finder(pruned_x, correcty, dataset_par, in_weight=pruned_w)

final_dict = combine_dict_list(miss_dict, corr_dict)

rule_limits = rule_limits_calculator(pruned_x, correcty, final_dict, alpha=alpha)

insignificant_neurons = [key for key, value in column_dict.items() if value not in sig_cols]
X_test, _ = input_delete(insignificant_neurons, X_test)
X_train, _ = input_delete(insignificant_neurons, X_train)
X_val, _ = input_delete(insignificant_neurons, X_val)

if len(rule_limits) > 1:
    rule_limits, rule_accuracy = rule_pruning(X_train, y_train, rule_limits, n_classes)
print(rule_limits)
print(rule_accuracy)

final_rules = rule_evaluator(X_val, y_val, rule_limits, rule_accuracy, np.unique(y))

num_test_examples = X_test.shape[0]
perturbed_data = perturbator(X_test)
rule_labels = np.empty(num_test_examples)
rule_labels[:] = np.nan
perturbed_labels = np.empty(num_test_examples)
perturbed_labels[:] = np.nan
for key, rule in final_rules.items():
    rule_labels = rule_elicitation(X_test, rule_labels, rule, key)
    perturbed_labels = rule_elicitation(perturbed_data, perturbed_labels, rule, key)

perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_classes + 10

completeness = sum(~np.isnan(rule_labels)) / num_test_examples

avg_length, number_rules = rule_size_calculator(final_rules)

rule_metrics_calculator(num_test_examples, y_test, rule_labels, predicted_labels, perturbed_labels, number_rules,
                        completeness, avg_length)
