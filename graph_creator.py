from mysql_queries import mysql_queries_executor
import pandas as pd
from common_functions import load_list


# method_list = ['NEW_METHOD_', 'RxNCM_', 'RxREN_']
method_list = ['NEW_METHOD_']
parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
data_path = 'datasets-UCI/new_rules/graph_files/'
save_graph = True

for df in [1, 3, 4]:
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
        print('-------------------------------------- FINAL RULES --------------------------------------')
        print(final_rules)
        print('-----------------------------------------------------------------------------------------')
        feature_set_name = 'NM' + dataset_par['dataset'] + "_ftrset"
        graph_name = 'NM' + dataset_par['dataset'] + "_grp"
        if save_graph:
            mysql_queries_executor(ruleset=final_rules, attacks=attack_list, conclusions=labels,
                                   feature_set_name=feature_set_name, graph_name=graph_name)
