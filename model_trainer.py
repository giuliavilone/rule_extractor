from common_functions import model_train
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from keras.utils import to_categorical
from common_functions import create_model
import sys


parameters = pd.read_csv('datasets-UCI/Used_data/summary.csv')
print(parameters)
target_var = 'class'

le = LabelEncoder()
out_lst = []
for index, item in parameters.iterrows():
    dataset = pd.read_csv('datasets-UCI/Used_data/' + item['dataset'] + '.csv')
    print(item['dataset'])
    col_types = dataset.dtypes
    for index, value in col_types.items():
        if value == 'object':
            dataset[index] = le.fit_transform(dataset[index].tolist())
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
                'trained_model_' + item['dataset'] + '.h5',
                n_epochs=item['epochs'], batch_size=item['batch_size'])
    max_index = history.history['accuracy'].index(max(history.history['accuracy']))
    out_lst.append([item['dataset'], history.history['accuracy'][max_index],
                    history.history['val_accuracy'][max_index]]
                   )

out_df = pd.DataFrame(out_lst, columns=['dataset', 'accuracy', 'val_accuracy'])
out_df.to_csv('datasets-UCI/Used_data/accuracy.csv')
