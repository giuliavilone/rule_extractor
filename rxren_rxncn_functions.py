from sklearn.metrics import accuracy_score
import numpy as np
import copy
from common_functions import create_model, rule_elicitation


def prediction_reshape(prediction_list):
    """
    The model returns the probabilities of a sample belonging to each output class. This function transforms them
    in an array of integers with the indexes of the class with the highest probability.
    :param prediction_list: the output of a keras model
    :return: array of predicted class
    """
    if len(prediction_list[0]) > 1:
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


def model_pruned_prediction(insignificant_index, in_df, in_item, in_weight=None):
    """
    Return the output classes predicted by the pruned model.
    :param insignificant_index: list of the insignificant input features
    :param in_df: input instances to be classified by the model
    :param in_item: list of model's hyper-parameters
    :param in_weight: model's weights
    :return: numpy array with the output classes predicted by the pruned model
    """
    input_x, w = input_delete(insignificant_index, in_df, in_weight=in_weight)
    new_m = create_model(input_x, in_item['classes'], in_item['neurons'], in_item['optimizer'],
                         in_item['init_mode'], in_item['activation'], in_item['dropout_rate'])
    new_m.set_weights(w)
    ret = new_m.predict(input_x)
    ret = prediction_reshape(ret)
    return ret


def rule_pruning(train_x, train_y, rule_set, classes_n):
    """
    Pruning the rules by removing those that do not affect the accuracy of the whole ruleset once removed
    :param train_x: pandas dataframe containing the input dataset (either the training or validation one)
    :param train_y: list of the output classes of the samples in the input dataset
    :param rule_set: dictionary of rules extracted from the trained model
    :param classes_n: number of output classes
    :return: the pruned ruleset and its accuracy
    """
    orig_acc = {}
    for rule_number in range(len(rule_set)):
        rule_list = rule_set[rule_number]
        cls = rule_list['class']
        orig_acc[cls] = ruleset_accuracy(train_x, train_y, rule_list, classes_n)
        ix = 0
        while len(rule_list['columns']) > 1 and ix < len(rule_list['columns']):
            new_rule = {'class': cls, 'columns': [v for i, v in enumerate(rule_list['columns']) if i != ix],
                        'limits': [v for i, v in enumerate(rule_list['limits']) if i != ix]}
            new_acc = ruleset_accuracy(train_x, train_y, new_rule, classes_n)
            if new_acc >= orig_acc[cls]:
                rule_list = new_rule
                orig_acc[cls] = new_acc
            else:
                ix += 1
        rule_set[rule_number] = rule_list
    return rule_set, orig_acc


def ruleset_accuracy(x_arr, y_list, rule_set, classes):
    """
    Compute the accuracy of the input dataset
    :param x_arr: numpy array containing the input dataset (either the training or validation one)
    :param y_list: list of the output classes of the samples in the input dataset
    :param rule_set: dictionary of rules extracted from the trained model
    :param classes: number of output classes
    :return: accuracy score of the input ruleset
    """
    predicted_y = np.empty(x_arr.shape[0])
    predicted_y[:] = np.NaN
    indexes = rule_elicitation(x_arr, rule_set)
    predicted_y[indexes] = rule_set['class']
    # If the rules do not cover some areas of the input space and there are no inferences made for some input instances,
    # the corresponding entrances in the list of the inferences are still Nan, so they must be replaced with an integer
    # to calculate the accuracy score. The solution is to use the number of output classes increased by 10
    predicted_y[np.isnan(predicted_y)] = classes + 10
    ret = accuracy_score(y_list, predicted_y)
    return ret


def rule_size_calculator(rule_set):
    """
    Calculate the number of rules and the average number of rules' antecedents
    :param rule_set: dictionary of rules extracted from the trained model
    :return: average number of rules' antecedents, number of rules
    """
    avg = 0
    length = 0
    for key, value in rule_set.items():
        length += len(value)
        avg += 1
    avg = avg / length
    return avg, length


def rule_formatter(rule_dict):
    """
    Change the format of the input ruleset as a list of dictionaries. Each dictionary is a rule and contains: 1) the
    output class predicted by the rule (this is the rule conclusion), 2) the list of variables of the input dataset, and
    3) the ranges of the input variables that form the rule's conditions.
    :param rule_dict: dictionary of rules extracted from the trained model
    :return: list of dictionaries containing the conditions and conclusions of each rule
    """
    ret = []
    for key, values in rule_dict.items():
        new_rule = {'class': key, 'columns': [v['columns'] for v in values], 'limits': [v['limits'] for v in values]}
        ret.append(new_rule)
    return ret


def rule_sorter(rules, in_df, cols_name):
    """
    This function sort the rules according to the number of covered instances, in reverse order (from the highest to
    the lowest number of covered instances). The rationale is that a small rule might represent an exception to a bigger
    rule. If the second is applied after the first, the accuracy of the entire ruleset would be decreased as the
    predictions made by the small rule would be erased when the big rule is elicited.
    :param rules: dictionary of rules extracted from the trained model
    :param in_df: pandas dataframe containing the input dataset (either the training or validation one)
    :param cols_name: list of the names of the in_df dataframe
    :return: list of sorted rules
    """
    ret = []
    for rule in rules:
        number_covered_instances = 0
        for item in range(len(rule['limits'])):
            minimum = rule['limits'][item][0]
            maximum = rule['limits'][item][1]
            number_covered_instances += len(in_df[np.logical_and(in_df[:, rule['columns'][item]] >= minimum,
                                                                 in_df[:, rule['columns'][item]] <= maximum)]
                                            )
        rule['instances'] = number_covered_instances
        rule['columns'] = [cols_name[r] for r in rule['columns']]
        ret.append(rule)
    ret = sorted(ret, key=lambda i: i['instances'], reverse=True)
    ret = [{k: v for k, v in d.items() if k != 'instances'} for d in ret]
    return ret
