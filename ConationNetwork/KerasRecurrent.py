from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
import keras as keras
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import sklearn.model_selection as sk
import pandas as pd


##############################################################################
data_dim = 10
timesteps = 10
dropout = 0.25
epochs = 25
batchSize = 64
validationSplit = 0.15
classes = 2
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, decay=1e-6)
model_path = ''
env_name = 'ConationModel'
##############################################################################


def load_data(label_name='ConationLevel'):

    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel', 'PredictedConation']


    train_path = "CombinedData_Data2.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    train_features, test_features = sk.train_test_split(train, test_size=0.20, random_state=42)
    train_features = train_features.drop(['ConationLevel'], axis=1)
    test_features = test_features.drop(['ConationLevel'], axis=1)
    train_features = train_features.drop(['PredictedConation'], axis=1)
    test_features = test_features.drop(['PredictedConation'], axis=1)

    train_label, test_label = sk.train_test_split(train.pop(label_name), test_size=0.20, random_state=42)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)


(train_feature, train_label), (test_feature, test_label) = load_data()

X_train = train_feature.values
X_test = test_feature.values


X_train = np.reshape(X_train, X_train.shape + (1,))
X_test = np.reshape(X_test, X_test.shape + (1,))

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=X_train.shape[1:]))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(X_train, train_label, batch_size=batchSize, epochs=epochs)

loss_and_metrics = model.evaluate(X_test, test_label, batch_size=batchSize)
print("\n" + "Loss: " + str(loss_and_metrics[0]) + "\n" + "Accuracy: " + str(loss_and_metrics[1]*100) + "%")
model.save('ConationModel_Stacked_LSTM.HDF5')