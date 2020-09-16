from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from scipy.stats import mode
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample


def create_model(train_x, num_classes, hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=train_x.shape[1], activation="sigmoid"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


def model_train(train_x, train_y, test_x, test_y, model, model_name, n_epochs=100, batch_size=10):
    check_pointer = ModelCheckpoint(filepath=model_name,
                                   save_weights_only=False,
                                   monitor='accuracy',
                                   save_best_only=True,
                                   verbose=1)
    history = model.fit(train_x, train_y, validation_data=(test_x, test_y), batch_size=batch_size,
              epochs=n_epochs, callbacks=[check_pointer]
              )
    return model, history


def perturbator(indf, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :type indf: Pandas dataframe
    """
    noise = np.random.normal(mu, sigma, indf.shape)
    return indf + noise


# make an ensemble prediction for multi-class classification
def ensemble_predictions(members, testX):
    # make predictions
    yhats = [model.predict(testX) for model in members]
    yhats = np.array(yhats)
    # combining the members via plurality voting
    voted_yhats = np.argmax(yhats, axis=2)
    results = mode(voted_yhats, axis=0)[0]
    return results


def dataset_uploader(item, target_var='class'):
    discrete_var = ['checking_status', 'credit_history', 'purpose', 'savings_status', 'employment', 'personal_status',
                    'other_parties', 'property_magnitude', 'other_payment_plans', 'housing', 'job', 'own_telephone',
                    'foreign_worker'
                    ]
    le = LabelEncoder()
    dataset = pd.read_csv('datasets-UCI/Used_data/' + item['dataset'] + '.csv')
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
    return X_train, X_test, y_train, y_test
