from scipy.io import arff
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense

data = arff.loadarff('datasets-UCI/UCI/iris.arff')
data = pd.DataFrame(data[0])
print(data.head())

# Separating independent variables from the target one
X = data.drop(columns=['class'])
le = LabelEncoder()
y = le.fit_transform(data['class'].tolist())
y = to_categorical(y, num_classes=3)

# define model
model = Sequential()
model.add(Dense(25, input_dim=2, activation='relu'))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(trainX, trainy, validation_data=(testX, testy), epochs=50, verbose=1)
