from sklearn.model_selection import cross_val_score
import sklearn.model_selection as sk
import pandas as pd
from sklearn import svm
import pickle as pickle

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


clf = svm.SVC()
print("Training")
clf.fit(train_feature, train_label)
#cross_val_score(clf, train_feature, train_label, scoring='accuracy')
print("Done Training")

filename = 'SVM_MODEL.sav'
pickle.dump(clf, open(filename, 'wb'))

