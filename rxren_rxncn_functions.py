from sklearn.metrics import accuracy_score
import numpy as np
import copy
from common_functions import create_model, rule_elicitation


def prediction_reshape(prediction_list):
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
    Calculate the output classes predicted by the pruned model.
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
    Pruning the rules
    :param train_x:
    :param train_y:
    :param rule_set:
    :param classes_n:
    :return:
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
    predicted_y = np.empty(x_arr.shape[0])
    predicted_y[:] = np.NaN
    indexes = rule_elicitation(x_arr, rule_set)
    predicted_y[indexes] = rule_set['class']
    predicted_y[np.isnan(predicted_y)] = classes + 10
    ret = accuracy_score(y_list, predicted_y)
    return ret


def rule_size_calculator(rule_set):
    avg = 0
    length = 0
    for key, value in rule_set.items():
        length += len(value)
        avg += 1
    avg = avg / length
    return avg, length


def rule_formatter(rule_dict):
    ret = []
    for key, values in rule_dict.items():
        new_rule = {'class': key, 'columns': [v['columns'] for v in values], 'limits': [v['limits'] for v in values]}
        ret.append(new_rule)
    return ret


def rule_sorter(rules, in_df, cols_name):
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
