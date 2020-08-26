from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import numpy as np


def create_model(train_x, num_classes, hidden_nodes):
    model = Sequential()
    model.add(Dense(hidden_nodes, input_dim=train_x.shape[1], activation="sigmoid"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    return model


def model_trainer(train_x, train_y, test_x, test_y, model, model_name, n_epochs=100):
    check_pointer = ModelCheckpoint(filepath=model_name,
                                   save_weights_only=False,
                                   monitor='loss',
                                   save_best_only=True,
                                   verbose=1)
    model.fit(train_x, train_y, validation_data=(test_x, test_y), epochs=n_epochs, callbacks=[check_pointer])
    return model


def perturbator(indf, mu=0, sigma=0.1):
    """
    Add white noise to input dataset
    :type indf: Pandas dataframe
    """
    noise = np.random.normal(mu, sigma, indf.shape)
    return indf + noise
