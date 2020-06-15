from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
from scipy.io import arff
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import sys

from trepan import Trepan,Oracle

n_cross_val = 10
n_class = 2

###########################################

def create_model (trainX,trainY,num_classes):
    model = Sequential()
    model.add(Dense(16, input_dim=trainX.shape[1], activation="sigmoid"))
    model.add(Dense(16, activation="sigmoid"))
    model.add(Dense(num_classes, activation="softmax"))
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(trainX, trainY, epochs=50, batch_size=10) # epochs=150
    return model

def vote_db_modifier(indf):
    indf.replace(b'y', 1, inplace=True)
    indf.replace(b'n', 0, inplace=True)
    indf.replace(b'?', 0, inplace=True)
    return indf


data = arff.loadarff('datasets-UCI/UCI/vote.arff')

data = pd.DataFrame(data[0])
X = data.drop(columns=['physician-fee-freeze','Class'])
le = LabelEncoder()
y = le.fit_transform(data['Class'].tolist())

# Replacing yes/no answers with 1/0
X = vote_db_modifier(X)

# Create the object to perform cross validation
skf = StratifiedKFold(n_splits=n_cross_val, random_state=7, shuffle=True)


fold_var = 1
#build oracle
for train_index, val_index in skf.split(X,y):
    print('Working on model number '+ str(fold_var))
    X_train, X_test = X[X.index.isin(train_index)], X[X.index.isin(val_index)]
    y_train, y_test = y[train_index], y[val_index]
    X_train = X_train.to_numpy()
    X_test = X_test.to_numpy()
    y_train = to_categorical(y_train, num_classes=n_class)
    model = create_model(X_train, y_train, n_class)
    oracle = Oracle(model, n_class, X_train)

    # build tree with TREPAN
    MIN_EXAMPLES_PER_NODE = 30
    MAX_NODES = 200
    root = Trepan.build_tree(MIN_EXAMPLES_PER_NODE, MAX_NODES, X_train, oracle)
    # calculate fidelity
    num_test_examples = X_test.shape[0]
    correct = 0
    for i in range(0, num_test_examples):
        ann_prediction = oracle.get_oracle_label(X_test[i, :])
        tree_prediction = root.classify(X_test[i, :])
        correct += (ann_prediction == tree_prediction)

    fidelity = float(correct) / num_test_examples

    print("Fidelity of the model is : " + str(fidelity))
    fold_var += 1