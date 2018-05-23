import keras as keras
import pandas as pd
import numpy as np
import DataVisualization as plot


###################################################################################

# Name of data file to load and make predictions on
data_file_name = "Data10_9.csv"

# Name of output file
output_file_name = "Predictions_P11.csv"

# Sets how wide the plot is. Higher is wider
Aspect = 4

#Show conationLevels, only possible on some data
show_Conation = False

#Path to original file
OriginalFile = 'Data10_9.txt'

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

model = keras.models.load_model(r'ConationModel.HDF5')

(Data_features, Data_labels) = load_data_one_set()

Predictions = model.predict_classes(Data_features, batch_size=None, verbose=0, steps=None)

"""
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
"""

output_df = pd.DataFrame(Predictions)
output_df.to_csv(output_file_name, index=False)

plot.plot(data_file_name, output_file_name, Aspect, show_Conation, OriginalFile)
