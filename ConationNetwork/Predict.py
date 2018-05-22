import keras as keras
import pandas as pd
import numpy as np



def load_data_one_set(label_name='ConationLevel'):
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel', 'PredictedConation'
                        ,'GameState', 'TimeSinceStart']

    train_path = "Data04_9.csv"

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

model = keras.models.load_model(r'C:\Users\dines\Desktop\Projects\ML-Conation\ConationNetwork\ConationModel.HDF5')

(predict_features, predict_labels) = load_data_one_set()

ymax = model.predict_classes(predict_features, batch_size=None, verbose=0, steps=None)
#ymax = model.evaluate(predict_features, predict_labels, batch_size=128)
predict_labels = predict_labels.astype(np.int32)
truePred = 0
falsePred = 0

print(np.sum(ymax))
print(np.sum(predict_labels))

for i in range(len(ymax)):
    if predict_labels[i] == ymax[i][0]:
        truePred +=1
    else:
        falsePred +=1

print("True: " + str(truePred))
print("False: " + str(falsePred))
print("Accuracy: " + str(truePred/(truePred+falsePred)))

