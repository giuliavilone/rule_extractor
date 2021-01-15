import pandas as pd
from common_functions import create_model, model_train, perturbator, dataset_uploader, rule_metrics_calculator
from keras.models import load_model
from refne import refne_run
from c45_test import run_c45_pane
from rxncn import rxncn_run
from rxren import rxren_run
from trepan_run import run_trepan
import sys

parameters = pd.read_csv('datasets/summary_new2.csv')
label_col = 'class'

for df in range(len(parameters)):
    metric_list = []
    dataset_par = parameters.iloc[df]
    print('--------------------------------------------------')
    print(dataset_par['dataset'])
    print('--------------------------------------------------')
    X_train, X_test, y_train, y_test, discrete_attributes, continuous_attributes = dataset_uploader(dataset_par,
                                                                                                    'datasets/',
                                                                                                    apply_smothe=False
                                                                                                    )
    X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]
    model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                       + str(dataset_par['best_model']) + '.h5')

    print('---------------------- Working on REFNE -----------------------')
    metric_refne = refne_run(X_train, X_test, y_train, y_test, discrete_attributes, continuous_attributes, label_col,
                             dataset_par, model
                             )
    metric_list.append(['REFNE'] + metric_refne)

    print('---------------------- Working on C45 PANE -----------------------')
    metric_c45 = run_c45_pane(X_train, X_test, y_test, dataset_par, model)
    metric_list.append(['C45 PANE'] + metric_c45)

    print('---------------------- Working on RXNCN -----------------------')
    metric_rxncn = rxncn_run(X_train, X_test, y_train, y_test, dataset_par, model)
    metric_list.append(['RXNCN'] + metric_rxncn)

    print('---------------------- Working on RXREN -----------------------')
    metric_rxren = rxren_run(X_train, X_test, y_train, y_test, dataset_par, model)
    metric_list.append(['RXREN'] + metric_rxren)

    print('---------------------- Working on TREPAN -----------------------')
    metric_trepan = run_trepan(X_train, X_test, y_train, y_test, discrete_attributes, dataset_par, model)
    metric_list.append(['TREPAN'] + metric_trepan)

    pd.DataFrame(metric_list, columns=['method', 'complete', 'correctness', 'fidelity', 'robustness', 'rule_n',
                                       'avg_length', 'overlap', 'class_fraction']
                 ).to_csv('metrics/metrics_' + dataset_par['dataset'] + '.csv')
