from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
import numpy as np
import sklearn.model_selection as sk
import pandas as pd
from numpy import array
from numpy import random


##############################################################################
data_dim = 10
timesteps = 10
dropout = 0.25
epochs = 1
batchSize = 200
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
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel',
                        'PredictedConation']

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


def load_Train_Test_Data():
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel']

    CSV_COLUMN_NAMES_TEST = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                             'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                             'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel',
                             'PredictedConation', 'GameState', 'TimeSinceStart']

    train_path = "TrainData.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    train_feature = train.drop(['ConationLevel'], axis=1)
    train_feature = train_feature.iloc[0:-19]
    train_label = train.pop('ConationLevel')
    train_label = train_label.replace([1, 2, 3, 4], 0)
    train_label = train_label.replace([5, 6, 7], 1)
    train_label = train_label .iloc[0:-19]
    test_path = "TestData.csv"

    # Parse the local CSV file.
    test = pd.read_csv(filepath_or_buffer=test_path,
                       names=CSV_COLUMN_NAMES_TEST,
                       header=0, sep=',')

    test_feature = test.drop(['ConationLevel'], axis=1)
    test_feature = test_feature.drop(['PredictedConation'], axis=1)
    test_feature = test_feature.drop(['GameState'], axis=1)
    test_feature = test_feature.drop(['TimeSinceStart'], axis=1)
    test_feature = test_feature.iloc[0:-186]

    test_label = test.pop('ConationLevel')
    test_label = test_label.replace([1, 2, 3, 4], 0)
    test_label = test_label.replace([5, 6, 7], 1)
    test_label = test_label.iloc[0:-186]
    return (train_feature, train_label), (test_feature, test_label)


(train_feature, train_label), (test_feature, test_label) = load_Train_Test_Data()

X_train = train_feature.values
Y_train = train_label.values
X_test = test_feature.values
Y_test = test_label.values

def slice_data(data, features, length):

    # split into samples (e.g. 5000/200 = 25)
    samples = list()
    length_timestep = 200
    n = data.shape
    # step over the 5,000 in jumps of 200
    for i in range(0, n[0], length_timestep):
        # grab from i to i + 200
        j = 0
        sample = data[i:i + length_timestep]
        print(sample[0][0])
        samples.append(sample)
        #samples[j] = sample
        j +=1

    returndata = array(samples)
    returndata = returndata.reshape((len(samples), length_timestep, features))
    return returndata

def slice_data_Y(data, features, length):

    # split into samples (e.g. 5000/200 = 25)
    samples = [0]*(length)
    length_timestep = 200
    n = data.shape
    # step over the 5,000 in jumps of 200
    for i in range(0, n[0], length_timestep):
        # grab from i to i + 200
        j = 0
        sample = data[i:i + length_timestep]
        print(len(sample))
        samples[j] = sample
        j +=1

    returndata = array(samples)
    returndata = returndata.reshape((len(samples), features))
    return returndata


X_train = slice_data(X_train, 10, 4862)
Y_train = slice_data_Y(Y_train, 1, 4862)
X_test = slice_data(X_test, 10, 865)
Y_test = slice_data_Y(Y_test, 1, 865)

print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)

print(type(X_train))
print(type(X_test))
print(type(Y_train))
print(type(Y_test))

def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 200, 10))
 batch_labels = np.zeros((200, 1))
 features = features.values
 labels = labels.values

 while True:
   for i in range(batch_size):
     # choose random index in features
     index = random.choice(len(features), 1)
     batch_features[i] = features[i]
     batch_labels[i] = labels[i]
   yield batch_features, batch_labels


# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(LSTM(32, return_sequences=True,
               input_shape=(200, 10)))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32, return_sequences=True))  # returns a sequence of vectors of dimension 32
model.add(LSTM(32))  # return a single vector of dimension 32
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#model.fit(X_train, Y_train, batch_size=64, epochs=epochs)
model.fit_generator(generator(train_feature, train_label, 200), steps_per_epoch=200, epochs=1)

loss_and_metrics = model.evaluate_generator(generator(test_feature, test_label, 200), steps=200)

#loss_and_metrics = model.evaluate(X_test, Y_test, batch_size=64)
print("\n" + "Loss: " + str(loss_and_metrics[0]) + "\n" + "Accuracy: " + str(loss_and_metrics[1] * 100) + "%")
model.save('ConationModel_Stacked_LSTM.HDF5')
