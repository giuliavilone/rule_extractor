from sklearn.metrics import accuracy_score
import numpy as np


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


def rule_elicitation(x, pred_y, rule_list, cls, over_y=None):
    for item in rule_list:
        minimum = item['limits'][0]
        maximum = item['limits'][1]
        if over_y is not None:
            indexes = np.where((x[:, item['neuron']] >= minimum) * (x[:, item['neuron']] <= maximum))[0]
            over_y += [x for x in indexes if not np.isnan(pred_y[x])]
        pred_y[(x[:, item['neuron']] >= minimum) * (x[:, item['neuron']] <= maximum)] = cls
    return pred_y, over_y


def ruleset_accuracy(x_arr, y_list, rule_set, cls, classes):
    predicted_y = np.empty(x_arr.shape[0])
    predicted_y[:] = np.NaN
    predicted_y, _ = rule_elicitation(x_arr, predicted_y, rule_set, cls)
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
