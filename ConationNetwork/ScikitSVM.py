from sklearn.model_selection import cross_val_score
import sklearn.model_selection as sk
import pandas as pd
from sklearn import svm
import pickle as pickle

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


clf = svm.SVC()

clf.fit(train_feature, train_label)


filename = 'SVM_MODEL.sav'
pickle.dump(clf, open(filename, 'wb'))

