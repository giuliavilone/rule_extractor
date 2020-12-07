import pandas as pd
from keras.models import load_model
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
import numpy as np
from sklearn.metrics import accuracy_score
import copy
from common_functions import perturbator, model_train, dataset_uploader, create_model, rule_metrics_calculator
from rxren_rxncn_functions import rule_pruning, rule_elicitation, ruleset_accuracy, rule_size_calculator
from scipy.stats import mode
np.random.seed(3)
import sys

# Functions
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


def prediction_reshape(prediction_list):
    if len(prediction_list[0]) > 1:
        ret = np.argmax(prediction_list, axis=1)
    else:
        ret = np.reshape(prediction_list, -1).tolist()
        ret = [round(x) for x in ret]
    return ret


def model_pruned_prediction(insignificant_index, in_df, in_item, in_weight=None):
    """
    Calculate the output classes predicted by the pruned model.
    :param insignificant_index: list of the insignificant input features
    :param in_df: input instances to be classified by the model
    :param in_item: list of model's hyper-parameters
    :param in_weight: model's weights
    :return: numpy array with the output classes predicted by the pruned model
    """
    input_x, w = input_delete(insignificant_index, in_df, in_weight=in_weight)
    new_m = create_model(input_x, in_item['classes'], in_item['neurons'], in_item['optimizer'],
                         in_item['init_mode'], in_item['activation'], in_item['dropout_rate'],
                         in_item['weight_constraint'], loss='categorical_crossentropy', out_activation='softmax')
    new_m.set_weights(w)
    ret = new_m.predict(input_x)
    ret = prediction_reshape(ret)
    return ret


def network_pruning(w, correct_x, correct_y, test_x, test_y, accuracy, in_item=None):
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
    insignificant_neurons = []
    miss_classified_list = []
    error_list = []
    new_acc = accuracy
    for i in range(w[0].shape[0]):
        res = model_pruned_prediction(i, correct_x, in_item, in_weight=w)
        misclassified = [i for i in range(len(correct_y)) if res[i] != correct_y[i]]
        miss_classified_list.append(misclassified)
        error_list.append(len(misclassified))
    # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
    # and its accuracy must be set equal to the accuracy of the original network
    if sum(error_list) > 0:
        pruning = True
    else:
        pruning = False

    theta = min(error_list)
    max_error = max(error_list)
    while pruning:
        insignificant_neurons = [i for i, e in enumerate(error_list) if e <= theta]
        new_res = model_pruned_prediction(insignificant_neurons, test_x, in_item, in_weight=w)
        new_acc = accuracy_score(new_res, test_y)
        if new_acc >= accuracy - TOLERANCE:
            new_error_list = [e for i, e in enumerate(error_list) if e > theta]
            # Leaving at least one significant neuron and avoid going up to the max error in case more than two
            # neurons share the same high error
            if len(new_error_list) > 1 and min(new_error_list) < max_error:
                theta = min(new_error_list)
            else:
                pruning = False
        else:
            pruning = False
    return miss_classified_list, sorted(insignificant_neurons), new_acc, error_list


def rule_limits_calculator(c_x, c_y, miss_classified_list, significant_neurons, error, alpha=0.1):
    c_tot = np.column_stack((c_x, c_y))
    grouped_miss_class = []
    for i in significant_neurons:
        miss_class = c_tot[miss_classified_list[i]]
        # Splitting the misclassified input values according to their output classes
        grouped_miss_class = grouped_miss_class + [{'neuron': i, 'class': k,
                                                    'limits': [min(miss_class[:, i][miss_class[:, -1] == k]),
                                                               max(miss_class[:, i][miss_class[:, -1] == k])]}
                                                   for k in np.unique(miss_class[:, -1])
                                                   if len(miss_class[:, i][miss_class[:, -1] == k]) > (error[i] * alpha)
                                                   ]
    # If no rules can be created, the rule set must assign the majority class
    if len(grouped_miss_class) == 0:
        grouped_miss_class += [{'neuron': 0, 'class': mode(c_y)[0][0], 'limits': [min(c_x[:, 0]), max(c_x[:, 0])]}]
    return grouped_miss_class


def rule_combiner(rule_set):
    ret = {}
    for i, r in enumerate(rule_set):
        if r['class'] not in ret.keys():
            ret[r['class']] = [{'neuron': r['neuron'], 'limits': r['limits']}]
        else:
            ret[r['class']].append({'neuron': r['neuron'], 'limits': r['limits']})
    return ret


def rule_evaluator(x, y, rule_dict, orig_acc, class_list):
    ret_rules = copy.deepcopy(rule_dict)
    rule_accuracy = copy.deepcopy(orig_acc)
    predicted_y = np.empty(x.shape[0])
    predicted_y[:] = np.NaN
    for cls, rule_list in rule_dict.items():
        predicted_y, _ = rule_elicitation(x, predicted_y, rule_list, cls)
    predicted_y[np.isnan(predicted_y)] = len(class_list) + 10
    # Calculate min and max of mismatched instances
    mismatched = [index for index, elem in enumerate(y) if elem != predicted_y[index]]
    for cls, rule_list in rule_dict.items():
        print('Working on class: ', cls)
        ixs = np.where(predicted_y == cls)[0].tolist()
        ixs = [ix for ix in ixs if ix in mismatched]
        if len(ixs) > 0:
            for pos in range(len(rule_list)):
                item = rule_list[pos]
                new_min = min(x[ixs, item['neuron']])
                new_max = max(x[ixs, item['neuron']])
                ret_rules[cls][pos] = {'neuron': item['neuron'], 'limits': [new_min, new_max]}
                new_acc = ruleset_accuracy(x, y, ret_rules[cls], cls, len(np.unique(y)))
                if new_acc < rule_accuracy[cls]:
                    ret_rules[cls][pos] = rule_dict[cls][pos]
                else:
                    rule_accuracy[cls] = new_acc
    return rule_accuracy, ret_rules


# Main code
# Global variable
TOLERANCE = 0.01
train_model = False

# Loading dataset
parameters = pd.read_csv('datasets-UCI/Used_data/summary_new.csv')
dataset_par = parameters.iloc[15]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('--------------------------------------------------')


MODEL_NAME = 'trained_model_' + dataset_par['dataset'] + '.h5'
X_train_list, X_test_list, y_train_list, y_test_list, _, _ = dataset_uploader(dataset_par)
metric_list = []
for ix in range(len(X_train_list)):
    X_train = X_train_list[ix]
    X_test = X_test_list[ix]
    y_train = y_train_list[ix]
    y_test = y_test_list[ix]
    y = np.concatenate((y_train, y_test), axis=0)
    X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
    n_classes = dataset_par['classes']

    if train_model:
        model = create_model(X_train, dataset_par['classes'], dataset_par['neurons'], eval(dataset_par['optimizer']),
                             dataset_par['init_mode'], dataset_par['activation'], dataset_par['dropout_rate'],
                             weight_constraint=eval(dataset_par['weight_constraint']), loss='binary_crossentropy',
                             out_activation='sigmoid'
                             )
        model_train(X_train, to_categorical(y_train, num_classes=dataset_par['classes']),
                    X_test, to_categorical(y_test, num_classes=dataset_par['classes']), model, MODEL_NAME,
                    n_epochs=dataset_par['epochs'], batch_size=dataset_par['batch_size']
                    )
    else:
        model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_' + str(ix) + '.h5')

    # model = load_model(MODEL_NAME)
    weights = np.array(model.get_weights())
    results = model.predict(X_train)
    results = np.argmax(results, axis=1)

    correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    print('Number of correctly classified examples', correctX.shape)
    correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
    acc = accuracy_score(results, y_train)
    print("Accuracy of original model on the train dataset: ", acc)
    test_pred = np.argmax(model.predict(X_test), axis=1)
    test_acc = accuracy_score(test_pred, y_test)
    print("Accuracy of original model on the test dataset: ", test_acc)

    miss_list, ins_index, new_accuracy, err = network_pruning(weights, correctX, correcty, X_test, y_test,
                                                              test_acc, dataset_par
                                                              )

    significant_index = [i for i in range(weights[0].shape[0]) if i not in ins_index]

    print("Accuracy of pruned network", new_accuracy)

    rule_limits = rule_limits_calculator(correctX, correcty, miss_list, significant_index, err, alpha=0.1)

    if len(rule_limits) > 1:
        rule_dict = rule_combiner(rule_limits)
        rule_limits, rule_acc = rule_pruning(X_test, y_test, rule_dict, n_classes)
    else:
        rule_limits = rule_combiner(rule_limits)
        rule_acc = {}
        for out_class, rule_list in rule_limits.items():
            rule_acc[out_class] = ruleset_accuracy(X_test, y_test, rule_list, out_class, len(np.unique(y)))

    rule_simplifier = True
    while rule_simplifier:
        new_rule_acc, final_rules = rule_evaluator(X_test, y_test, rule_limits, rule_acc, np.unique(y))
        if sum(new_rule_acc.values()) > sum(rule_acc.values()):
            rule_acc = new_rule_acc
        else:
            rule_simplifier = False

    print(final_rules)

    predicted_labels = model.predict(X_test)
    predicted_labels = np.argmax(predicted_labels, axis=1)
    num_test_examples = X_test.shape[0]
    perturbed_data = perturbator(X_test)
    rule_labels = np.empty(num_test_examples)
    rule_labels[:] = np.nan
    perturbed_labels = np.empty(num_test_examples)
    perturbed_labels[:] = np.nan
    overlap = []
    for key, rule in final_rules.items():
        rule_labels, overlap = rule_elicitation(X_test, rule_labels, rule, key, over_y=overlap)
        perturbed_labels, _ = rule_elicitation(perturbed_data, perturbed_labels, rule, key)

    perturbed_labels[np.where(np.isnan(perturbed_labels))] = n_classes + 10
    completeness = sum(~np.isnan(rule_labels)) / num_test_examples
    avg_length, number_rules = rule_size_calculator(final_rules)
    overlap = len(set(overlap)) / len(X_test)
    metric_list.append(rule_metrics_calculator(num_test_examples, y_test, rule_labels, predicted_labels,
                                               perturbed_labels, number_rules,
                                               completeness, avg_length, overlap, dataset_par['classes']
                                                )
                       )
pd.DataFrame(metric_list, columns=['complete', 'correctness', 'fidelity', 'robustness', 'rule_n', 'avg_length',
                                   'overlap', 'class_fraction']
             ).to_csv('rxren_metrics_' + dataset_par['dataset'] + '.csv')
