import pandas as pd
from keras.utils import to_categorical
from keras.optimizers import SGD, Adagrad, Adam, Nadam
from common_functions import create_model, model_train, perturbator, dataset_uploader, rule_metrics_calculator
from keras.models import load_model


parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
dataset_par = parameters.iloc[11]
print('--------------------------------------------------')
print(dataset_par['dataset'])
print('----------------------')
X_train, X_test, y_train, y_test, _, _ = dataset_uploader(dataset_par)
X_train, X_test = X_train.to_numpy(), X_test.to_numpy()
model = load_model('trained_model_' + dataset_par['dataset'] + '.h5')
n_class = dataset_par['classes']