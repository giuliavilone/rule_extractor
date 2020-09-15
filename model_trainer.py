from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.constraints import maxnorm
from common_functions import model_train
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
import sys


def create_model(train_x, n_classes, neurons, optimizer, init_mode, activation, dropout_rate, weight_constraint):
    # create model
    model = Sequential()
    model.add(Dense(neurons, input_dim=train_x.shape[1], activation=activation, kernel_initializer=init_mode,
                    kernel_constraint=maxnorm(weight_constraint)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(n_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model


discrete_var = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status',
                'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone',
                'foreign_worker'
                ]

parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
print(parameters)
target_var = 'class'

le = LabelEncoder()
out_lst = []
for index, item in parameters.iterrows():
    dataset = pd.read_csv('datasets-UCI/Used_data/' + item['dataset'] + '.csv')
    print(item['dataset'])
    if item['dataset'] == 'credit-g':
        for var in discrete_var:
            dataset[var] = le.fit_transform(dataset[var].tolist())
    # Separating independent variables from the target one
    X = dataset.drop(columns=[target_var]).to_numpy()
    y = le.fit_transform(dataset[target_var].tolist())
    ix = [i for i in range(len(X))]
    train_index = resample(ix, replace=True, n_samples=int(len(X) * 0.7))
    val_index = [x for x in ix if x not in train_index]
    X_train, X_test = X[train_index], X[val_index]
    y_train, y_test = y[train_index], y[val_index]
    # define model
    model = create_model(X, item['classes'], item['neurons'], item['optimizer'], item['init_mode'],
                         item['activation'], item['dropout_rate'], item['weight_constraint']
                         )
    _, history = model_train(X_train, to_categorical(y_train, num_classes=item['classes']),
                X_test, to_categorical(y_test, num_classes=item['classes']), model,
                'trained_model_' + item['dataset'] + '.h5')
    max_index = history.history['accuracy'].index(max(history.history['accuracy']))
    out_lst.append([item['dataset'], history.history['accuracy'][max_index],
                    history.history['val_accuracy'][max_index]]
                   )

out_df = pd.DataFrame(out_lst, columns=['dataset', 'accuracy', 'val_accuracy'])
out_df.to_csv('datasets-UCI/Used_data/accuracy.csv')
