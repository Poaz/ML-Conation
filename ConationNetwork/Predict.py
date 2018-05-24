import keras as keras
import pandas as pd
import numpy as np
import DataVisualization as plot


###################################################################################

# Name of data file to load and make predictions on
data_file_name = "Data10_9.txt"

# Name of output file
output_file_name = "Predictions_P11.csv"

# Sets how wide the plot is. Higher is wider
Aspect = 4

#Show conationLevels, only possible on some data
show_Conation = True

#Path to original file
OriginalFile = 'Data10_9.txt'

#Sets which nTH row to plot. Example: 50, is sampling every 50th row (1 sample/s)
resample_rate = 50

####################################################################################


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

(train_features, train_labels), (test_feature, test_label) = load_Train_Test_Data()

X_train = np.reshape(test_feature.values, (test_feature.values).shape + (1,))

Predictions = model.predict_classes(X_train , batch_size=64)

Data_labels = test_label.replace([1, 2, 3, 4], 0)
Data_labels = Data_labels.replace([5, 6, 7], 1)

Data_labels = Data_labels.astype(np.int32)
truePred = 0
falsePred = 0

print(np.sum(Predictions))
print(np.sum(Data_labels))

for i in range(len(Predictions)):
    if Data_labels[i] == Predictions[i][0]:
        truePred +=1
    else:
        falsePred +=1

print("True: " + str(truePred))
print("False: " + str(falsePred))
print("Accuracy: " + str(truePred/(truePred+falsePred)))


#output_df = pd.DataFrame(Predictions)
#output_df.to_csv(output_file_name, index=False)

#plot.plot(data_file_name, output_file_name, Aspect, show_Conation, OriginalFile, resample_rate)
