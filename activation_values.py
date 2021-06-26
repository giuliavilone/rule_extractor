from keras import backend as K
from keras.models import load_model
import pandas as pd
import numpy as np
from keras import models
from common_functions import dataset_uploader
import matplotlib.pyplot as plt
import seaborn as sns


parameters = pd.read_csv('datasets-UCI/new_rules/summary.csv')
data_path = 'datasets-UCI/new_rules/'
dataset_par = parameters.iloc[3]
label_col = dataset_par['output_name']
model = load_model('trained_models/trained_model_' + dataset_par['dataset'] + '_'
                   + str(dataset_par['best_model']) + '.h5'
                   )
print(model.summary())

inp = model.input                                           # input placeholder
outputs = [layer.output for layer in model.layers][1:]        # all layer outputs except first (input) layer

activation_model = models.Model(inputs=model.input, outputs=outputs)


X_train, X_test, y_train, y_test, labels, disc_attributes, cont_attributes = dataset_uploader(dataset_par,
                                                                                                  data_path,
                                                                                                  target_var=label_col,
                                                                                                  cross_split=3,
                                                                                                  apply_smothe=False
                                                                                                  )

X_train, X_test, y_train, y_test = X_train[0], X_test[0], y_train[0], y_test[0]
activations = activation_model.predict(X_test)
print(X_test.shape)
print(len(activations))
print(activations[0].shape)
print(activations[1].shape)
print(activations[2].shape)
print(activations[3].shape)

sns.heatmap(activations[0])
plt.show()
sns.heatmap(activations[1])
plt.show()
sns.heatmap(activations[2])
plt.show()