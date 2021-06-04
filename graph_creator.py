from mysql_queries import mysql_queries_executor
import pandas as pd
from common_functions import load_list


method_list = ['NEW_METHOD_', 'RxNCM_', 'REFNE_', 'RxREN_']
parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
data_path = 'datasets-UCI/new_rules/'

for df in range(len(parameters)):
    dataset_par = parameters.iloc[df]
    labels = load_list(dataset_par['dataset'] + "_labels")
    for method in method_list:
        attack_list = load_list(method + dataset_par['dataset'] + "_attack_list")
        final_rules = load_list(method + dataset_par['dataset'] + "_final_rules")
        feature_set_name = method + dataset_par['dataset'] + "_featureset"
        graph_name = method + dataset_par['dataset'] + "_graph"
        mysql_queries_executor(ruleset=final_rules, attacks=attack_list, conclusions=labels,
                               feature_set_name=feature_set_name, graph_name=graph_name)
