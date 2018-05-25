from numpy import random

from keras.layers import Conv1D, GlobalAveragePooling1D, MaxPooling1D, MaxPooling2D
import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD, Adam
from pip.req.req_file import process_line


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

    train_label = train.pop('ConationLevel')
    train_label = train_label.replace([1, 2, 3, 4], 0)
    train_label = train_label.replace([5, 6, 7], 1)

    test_path = "TestData.csv"

    # Parse the local CSV file.
    test = pd.read_csv(filepath_or_buffer=test_path,
                        names=CSV_COLUMN_NAMES_TEST,
                        header=0, sep=',')

    test_feature = test.drop(['ConationLevel'], axis=1)
    test_feature = test_feature.drop(['PredictedConation'], axis=1)
    test_feature = test_feature.drop(['GameState'], axis=1)
    test_feature = test_feature.drop(['TimeSinceStart'], axis=1)

    test_label = test.pop('ConationLevel')
    test_label = test_label.replace([1, 2, 3, 4], 0)
    test_label = test_label.replace([5, 6, 7], 1)


    return(train_feature, train_label), (test_feature, test_label)


(train_feature, train_label), (test_feature, test_label) = load_Train_Test_Data()


def generator(features, labels, batch_size):
 # Create empty arrays to contain batch of features and labels#
 batch_features = np.zeros((batch_size, 200, 10))
 batch_labels = np.zeros((batch_size, 200, 1))
 features = features.values
 labels = labels.values

 while True:
   for i in range(batch_size):
     # choose random index in features
     index= random.choice(len(features), 1)
     batch_features[i] = features[index]
     batch_labels[i] = labels[index]
   yield batch_features, batch_labels


model = Sequential()

model.add(Conv1D(32, 3, input_shape=(200, 10), activation='sigmoid',  padding='same'))
model.add(Dropout(0.25))
model.add(Conv1D(16, 3, activation='sigmoid',  padding='same'))
model.add(Dropout(0.20))
model.add(Dense(30, activation='sigmoid'))
model.add(Dropout(0.15))
model.add(Dense(14, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.fit_generator(generator(train_feature, train_label, 200),
                    steps_per_epoch=200, epochs=10)

loss_and_metrics = model.evaluate_generator(generator(test_feature, test_label, 200), steps=200)
print("\n" + "Loss: " + str(loss_and_metrics[0]) + "\n" + "Accuracy: " + str(loss_and_metrics[1]*100) + "%")
#model.fit(train_feature, train_label, batch_size=16, epochs=10)
#score = model.evaluate(test_feature, test_label, batch_size=16)