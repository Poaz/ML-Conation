import keras as keras
import pandas as pd
import numpy as np
import DataVisualization as plot
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import itertools
import time



###################################################################################

# Name of data file to load and make predictions on
data_file_name = "TestData\Data24_8.txt"

# Name of output file
output_file_name = "Predictions_P11.csv"

# Sets how wide the plot is. Higher is wider
Aspect = 4

#Show conationLevels, only possible on some data
show_Conation = False

#Path to original file
OriginalFile = 'TestData\Data24_8.txt'

#Sets which nTH row to plot. Example: 50, is sampling every 50th row (1 sample/s)
resample_rate = 50

class_names = ['Low', 'High']

####################################################################################
class MainGenerator(object):

    def __init__(self, features, labels, batch_size):
        self.features = features
        self.labels = labels
        self.batch_size = batch_size
        self.currentStep = 0

    def generator(self):

        features = self.features
        labels = self.labels
        # Create empty arrays to contain batch of features and labels#
        batch_features = np.zeros((self.batch_size, 200, 10))
        batch_labels = np.zeros((200, 1))
        features = features.values
        labels = labels.values
        while True:
            for i in range(0, self.batch_size):
                batch_features[i] = features[i+self.currentStep]
                batch_labels[i] = labels[i+self.currentStep]
                self.currentStep += 1
                if((features.shape[0]-200) == self.currentStep):
                    self.currentStep = 0
            yield batch_features, batch_labels


def load_data_one_set(label_name='ConationLevel'):
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel', 'PredictedConation'
                        ,'GameState', 'TimeSinceStart']

    train_path = data_file_name

    # Parse the local CSV file.
    data = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    dataset_features = data
    dataset_features = dataset_features.drop(['ConationLevel'], axis=1)
    dataset_features = dataset_features.drop(['PredictedConation'], axis=1)
    dataset_features = dataset_features.drop(['GameState'], axis=1)
    dataset_features = dataset_features.drop(['TimeSinceStart'], axis=1)

    dataset_labels = data.pop(label_name)

    return (dataset_features, dataset_labels)

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


model = keras.models.load_model(r'ConationModel_Stacked_LSTM.HDF5')

(test_feature, test_label) = load_data_one_set()

Predictions = model.predict_generator(MainGenerator(test_feature, test_label, 64).generator(), steps=100)

print(np.shape(Predictions))
BinaryLabels = np.zeros(np.shape(test_label))

for i in range(len(Predictions)):
    if Predictions[i] >= 0.5:
        BinaryLabels[i] = 1
    else:
        BinaryLabels[i] = 0

truePred = 0
falsePred = 0

for i in range(len(BinaryLabels)):
    if test_label[i] == BinaryLabels[i]:
        truePred +=1
    else:
        falsePred +=1

print("True: " + str(truePred))
print("False: " + str(falsePred))
print("Accuracy: " + str(truePred/(truePred+falsePred)))

output_df = pd.DataFrame(BinaryLabels)
output_df.to_csv(output_file_name, index=False)

plot.plot(data_file_name, output_file_name, Aspect, show_Conation, OriginalFile, resample_rate)

