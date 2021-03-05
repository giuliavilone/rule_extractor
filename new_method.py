import pandas as pd
import copy
import numpy as np
from keras.models import load_model
from rxren_rxncn_functions import input_delete, model_pruned_prediction
from common_functions import dataset_uploader
from refne import synthetic_data_generator
from sklearn import svm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys


def prediction_classifier(orig_y, predict_y, comparison="misclassified"):
    ret = {}
    out_classes = np.unique(orig_y)
    for cls in out_classes:
        if comparison == 'misclassified':
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] != orig_y[i] and orig_y[i] == cls]
        else:
            ret[cls] = [i for i in range(len(orig_y)) if predict_y[i] == orig_y[i] and orig_y[i] == cls]
    return ret


def network_pruning(w, correct_x, correct_y, in_item=None):
    """
    Remove the insignificant input neurons of the input model, based on the weight w.
    :param w: model's weights
    :param correct_x: set of correctly classified instances (independent variables)
    :param correct_y: set of correctly classified instances (dependent variable)
    :param test_x: test dataset (independent variables)
    :param in_item: in_item
    :return: miss-classified instances, pruned weights, accuracy of pruned model,
    """
    temp_w = copy.deepcopy(w)
    temp_x = copy.deepcopy(correct_x.to_numpy())
    significant_cols = correct_x.columns.tolist()
    miss_classified_dict = {}
    pruning = True
    while pruning:
        error_list = []
        miss_classified_dict = {}
        for i in range(temp_w[0].shape[0]):
            res = model_pruned_prediction(i, temp_x, in_item, in_weight=temp_w)
            misclassified = prediction_classifier(correct_y, res)
            miss_classified_dict[significant_cols[i]] = misclassified
            error_list.append(sum([len(value) for key, value in misclassified.items()]))
        # In case the pruned network correctly predicts all the test inputs, the original network cannot be pruned
        # and its accuracy must be set equal to the accuracy of the original network
        if min(error_list) == 0:
            insignificant_neurons_temp = [i for i, e in enumerate(error_list) if e == 0]
            significant_cols = [significant_cols[i] for i in significant_cols if i not in insignificant_neurons_temp]
            temp_x, temp_w = input_delete(insignificant_neurons_temp, temp_x, in_weight=temp_w)
        else:
            pruning = False
    return significant_cols


def remove_column(df, column_tbm, in_weight=None):
    """
    Remove the columns not listed in column_tbm from the input dataframe and, if not none, from the model's weights
    :param df: input pandas dataframe
    :param in_weight: array of model's weights
    :param column_tbm: list of columns to be maintained in the input dataframe and the array of weights
    :return:
    """
    for i, col in enumerate(df.columns.tolist()):
        if col not in column_tbm:
            df = df.drop(col, axis=1)
            if in_weight is not None:
                in_weight[0] = np.delete(in_weight[0], i, 0)
    return df, in_weight


parameters = pd.read_csv('datasets-UCI/UCI_csv/summary.csv')
label_col = 'class'
data_path = 'datasets-UCI/UCI_csv/'
dataset_par = parameters.iloc[0]
print(dataset_par['dataset'])
X_train, X_test, y_train, y_test, discrete_attributes, continuous_attributes = dataset_uploader(dataset_par,
                                                                                                data_path,
                                                                                                apply_smothe=False
                                                                                                )

X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]
model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                   + str(dataset_par['best_model']) + '.h5'
                   )

results = model.predict_classes(X_train)
correctX = X_train[[results[i] == y_train[i] for i in range(len(y_train))]]
correcty = y_train[[results[i] == y_train[i] for i in range(len(y_train))]]
weights = np.array(model.get_weights())
significant_features = network_pruning(weights, correctX, correcty, in_item=dataset_par)
X_train, weights = remove_column(X_train, significant_features, in_weight=weights)

xSynth = synthetic_data_generator(X_train, X_train.shape[0] * 2)
X = xSynth.append(X_train, ignore_index=True)
X = X.to_numpy()
y = np.argmax(model.predict(X), axis=1)

svc = svm.SVC(kernel='linear').fit(X[:, :3], y)
# print(dir(svc))
# print(svc.support_)
# print(svc.support_vectors_)

# The equation of the separating plane is given by all x so that np.dot(svc.coef_[0], x) + b = 0.
# Solve for w3 (z)
z = lambda x, y: (-svc.intercept_[0]-svc.coef_[0][0]*x -svc.coef_[0][1]*y) / svc.coef_[0][2]
tmp = np.linspace(0, 10, 10)
x,y = np.meshgrid(tmp, tmp)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 1], X[:, 2], X[:, 3])
ax.plot_surface(x, y, z(x, y))
ax.view_init(30, 60)
plt.show()
