# Code from https://github.com/marcotcr/anchor

from __future__ import print_function
import numpy as np
np.random.seed(1)
import sklearn
from sklearn.ensemble import RandomForestClassifier
from anchor import utils
from anchor import anchor_tabular

dataset_folder = 'datasets-UCI/'
dataset = utils.load_dataset('adult', balance=True, dataset_folder=dataset_folder, discretize=True)

c = RandomForestClassifier(n_estimators=50, n_jobs=5)
c.fit(dataset.train, dataset.labels_train)
print('Train', sklearn.metrics.accuracy_score(dataset.labels_train, c.predict(dataset.train)))
print('Test', sklearn.metrics.accuracy_score(dataset.labels_test, c.predict(dataset.test)))

explainer = anchor_tabular.AnchorTabularExplainer(
    dataset.class_names,
    dataset.feature_names,
    dataset.train,
    dataset.categorical_names
)

idx = 0
np.random.seed(1)
print('Prediction: ', explainer.class_names[c.predict(dataset.test[idx].reshape(1, -1))[0]])
exp = explainer.explain_instance(dataset.test[idx], c.predict, threshold=0.95)
print('Anchor: %s' % (' AND '.join(exp.names())))
print('Precision: %.2f' % exp.precision())
print('Coverage: %.2f' % exp.coverage())