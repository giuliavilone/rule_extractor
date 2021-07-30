import pandas as pd
from common_functions import save_list, create_empty_file, dataset_uploader
from keras.models import load_model
from refne import refne_run
from c45_test import run_c45_pane
from rxncn import rxncn_run
from rxren import rxren_run
from trepan_run import run_trepan
from new_method import cluster_rule_extractor


parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
data_path = 'datasets-UCI/new_rules/'
save_graph = True

for df in [0, 1, 3, 4]:
    metric_list = []
    dataset_par = parameters.iloc[df]
    label_col = dataset_par['output_name']
    print('--------------------------------------------------')
    print(dataset_par['dataset'])
    print('--------------------------------------------------')
    X_train, X_test, y_train, y_test, labels, disc_attributes, cont_attributes = dataset_uploader(dataset_par,
                                                                                                  data_path,
                                                                                                  target_var=label_col,
                                                                                                  cross_split=3,
                                                                                                  apply_smothe=False
                                                                                                  )
    create_empty_file(dataset_par['dataset'] + "_labels")
    save_list(labels, dataset_par['dataset'] + "_labels")

    X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]

    model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                       + str(dataset_par['best_model']) + '.h5')

    print('---------------------- Working on REFNE -----------------------')
    # metric_refne = refne_run(X_train, X_test, y_test, disc_attributes, cont_attributes, dataset_par, model, save_graph)
    # metric_list.append(['REFNE'] + metric_refne)

    print('---------------------- Working on C45 PANE -----------------------')
    # metric_c45 = run_c45_pane(X_train, X_test, y_test, dataset_par, model, labels)
    # metric_list.append(['C45 PANE'] + metric_c45)

    print('---------------------- Working on RxNCM -----------------------')
    # metric_rxncn = rxncn_run(X_train, X_test, y_train, y_test, dataset_par, model, save_graph)
    # metric_list.append(['RXNCM'] + metric_rxncn)

    print('---------------------- Working on RxREN -----------------------')
    # metric_rxren = rxren_run(X_train, X_test, y_train, y_test, dataset_par, model, save_graph)
    # metric_list.append(['RXREN'] + metric_rxren)

    print('---------------------- Working on TREPAN -----------------------')
    # metric_trepan = run_trepan(X_train, X_test, y_train, y_test, disc_attributes, dataset_par, model)
    # metric_list.append(['TREPAN'] + metric_trepan)

    print('---------------------- Working on NEW METHOD -----------------------')
    new_metrics = cluster_rule_extractor(X_train, X_test, y_train, y_test, dataset_par, save_graph)
    metric_list.append(['NEW METHOD'] + new_metrics)

    pd.DataFrame(metric_list, columns=['method', 'complete', 'correctness', 'fidelity', 'robustness', 'rule_n',
                                       'avg_length', 'overlap', 'class_fraction']
                 ).to_csv('metrics/metrics_' + dataset_par['dataset'] + '.csv')
