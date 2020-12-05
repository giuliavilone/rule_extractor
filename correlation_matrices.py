import pandas as pd
import glob
import seaborn as sn
import matplotlib.pyplot as plt
import copy
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import itertools
from sklearn.preprocessing import LabelEncoder
import sys


def correlation_matrix_plot(in_corr_matrix, save_corr_matrix=False):
    sns_plot = sn.heatmap(in_corr_matrix, annot=True)
    plt.title(dataset_name, fontsize=14)
    plt.xticks(rotation=18, horizontalalignment='right')
    plt.show()
    if save_corr_matrix:
        sns_plot = sns_plot.get_figure()
        sns_plot.savefig(dataset_name + '.png')


def feature_selector(in_corr_matrix, high_correlated=True, threshold=0.5):
    ret = copy.deepcopy(in_corr_matrix)
    new_corr_matrix = ret.replace(1, 0).abs()
    if high_correlated:
        corr_index = new_corr_matrix.index[new_corr_matrix.max(axis=1) < threshold]
    else:
        corr_index = new_corr_matrix.index[new_corr_matrix.max(axis=1) >= threshold]
    ret = ret.drop(corr_index, axis=0)
    ret = ret.drop(corr_index, axis=1)
    return ret


def fit_linear_reg(x, y):
    """
    Fit linear regression model and return RSS and R squared values
    :param x:
    :param y:
    :return:
    """
    model_k = linear_model.LinearRegression(fit_intercept=True)
    model_k.fit(x, y)
    rss = mean_squared_error(y, model_k.predict(x)) * len(y)
    r_squared = model_k.score(x, y)
    return rss, r_squared


def find_best_combination(in_df, y, corr_features):
    """
    Find the best combination of input features from the list of those that are highly correlated
    :param in_df:
    :param y:
    :param corr_features:
    :return:
    """
    rss_list, r_squared_list, feature_list, numb_features = [], [], [], []
    print(len(corr_features))
    for k in range(1, len(corr_features)):
        for combo in itertools.combinations(corr_features, k):
            tmp_result = fit_linear_reg(in_df[list(combo)], y)
            rss_list.append(tmp_result[0])
            r_squared_list.append(tmp_result[1])
            feature_list.append(combo)
            numb_features.append(len(combo))
    ret = pd.DataFrame({'numb_features': numb_features, 'RSS': rss_list, 'R_squared': r_squared_list,
                        'features': feature_list}
                       )
    return ret


def data_importer(file_to_be_imported, file_name, remove_columns=True):
    le = LabelEncoder()
    feat_to_be_deleted = {'bank': ['euribor3m', 'emp.var.rate'],
                          'cover type': ['Wilderness_Area1', 'Aspect', 'Hillshade_9am', 'Hor_Dist_Hydrology'],
                          'letter recognition': ['y-box', 'high', 'width'],
                          'online shoppers intention': ['BounceRates', 'ProductRelated', 'Inform_Duration'],
                          'avila': ['F10'],
                          'credit card default': ['BILL_AMT6', 'BILL_AMT5', 'BILL_AMT4', 'BILL_AMT3', 'BILL_AMT2',
                                                  'PAY_6', 'PAY_5', 'PAY_4', 'PAY_3', 'PAY_2'],
                          'eeg eye states': ['P7', 'F8', 'T8', 'P8', 'FC5'],
                          'skin nonskin': ['B'],
                          'htru': ['mean_dm_snr_curve', 'kurtosis_dm_snr_curve', 'skewness_profile', 'mean_profile'],
                          'occupancy': ['HumidityRatio', 'Temperature'],
                          'shuttle': ['S7', 'S8', 'S9']
                          }

    ret_df = pd.read_csv(file_to_be_imported)
    ret_class = le.fit_transform(ret_df['class'].tolist())
    ret_df = ret_df.drop(columns=['class'])
    if remove_columns:
        if file_name in feat_to_be_deleted.keys():
            columns_to_be_deleted = [item for item in ret_df.columns.tolist() if item in feat_to_be_deleted[file_name]]
            ret_df = ret_df.drop(columns=columns_to_be_deleted)
    col_types = ret_df.dtypes
    for index, value in col_types.items():
        if value in ['object', 'bool']:
            if index != 'class':
                ret_df[index] = le.fit_transform(ret_df[index].tolist())
    return ret_df, ret_class


path = "/home/d18126441/PycharmProjects/rule_extractor/datasets-UCI/new_datasets/*.csv"

plot_matrix = True
best_combination = False
for filename in glob.glob(path):
    dataset_name = filename[73:-4].replace("_", " ")
    print(dataset_name)
    df, out_class = data_importer(filename, dataset_name)

    corr_matrix = df.corr().round(4)
    if dataset_name in ('cover type', 'connect-4'):
        # The cover type dataset has too many variables to be plotted in a single matrix. So here the function
        # removes those that are weakly correlated with all the others and plots only those with
        # some correlation values that are greater than 0.5
        corr_matrix = feature_selector(corr_matrix)
    elif dataset_name in ('diabetic data'):
        corr_matrix = feature_selector(corr_matrix, threshold=0.3)
    if plot_matrix:
        correlation_matrix_plot(corr_matrix)
    # Finding the best combination among the correlated variables
    if best_combination:
        new_matrix = feature_selector(corr_matrix, threshold=0.6)
        if len(new_matrix) > 0:
            high_corr_feat = new_matrix.columns.tolist()
            best_feat = find_best_combination(df, out_class, high_corr_feat)
            best_feat.to_csv('best_feat_' + dataset_name + '.csv')
