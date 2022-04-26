from mysql_queries import mysql_queries_executor
import pandas as pd
from common_functions import load_list, dataset_uploader
import numpy as np
from keras.models import load_model


def data_selector(rule_list, max_sample_number=10):
    ret = []
    for rule in rule_list:
        sample_list = rule['samples']
        sample_list = np.random.choice(sample_list, min(len(sample_list), max_sample_number))
        ret += sample_list.tolist()
    return ret


# method_list = ['NEW_METHOD_', 'RxNCM_', 'RxREN_']
method_list = ['NEW_METHOD_']
parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
data_path = 'datasets-UCI/new_rules/graph_files/'
save_graph = True

for df in [0]:
    dataset_par = parameters.iloc[df]
    print('-----------------------------------------------------------------------------------------')
    print("Dataset: ", dataset_par['dataset'])
    labels = load_list(dataset_par['dataset'] + "_labels", data_path)
    for method in method_list:
        print("Method: ", method)
        attack_list = load_list(method + dataset_par['dataset'] + "_attack_list", data_path)
        print('-------------------------------------- ATTACK LIST --------------------------------------')
        print(attack_list)
        print('-----------------------------------------------------------------------------------------')
        final_rules = load_list(method + dataset_par['dataset'] + "_final_rules", data_path)
        # print('-------------------------------------- FINAL RULES --------------------------------------')
        # print(final_rules)
        print('-----------------------------------------------------------------------------------------')
        sample_indexes = data_selector(final_rules)
        X_train, X_test, y_train, y_test, _, _, _ = dataset_uploader(dataset_par['dataset'],
                                                                     'datasets-UCI/new_rules/',
                                                                     target_var=dataset_par['output_name'],
                                                                     apply_smote=False,
                                                                     data_normalization=False)
        X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]
        model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                           + str(int(dataset_par['best_model'])) + '.h5')
        X_tot = pd.concat([X_train, X_test], ignore_index=True)
        y_tot = np.concatenate((y_train, y_test))
        y_pred = np.argmax(model.predict(X_tot), axis=1)
        X_tot['Observed class'] = y_tot.tolist()
        X_tot['Model predicted class'] = y_pred
        X_tot = X_tot.iloc[sample_indexes]
        X_tot.to_csv('short_data.csv')
        if save_graph:
            feature_set_name = dataset_par['dataset'] + "_ftrset"
            graph_name = dataset_par['dataset'] + "_grp"
            mysql_queries_executor(database=dataset_par['database'], ruleset=final_rules, attacks=attack_list,
                                   conclusions=labels, feature_set_name=feature_set_name, graph_name=graph_name)
