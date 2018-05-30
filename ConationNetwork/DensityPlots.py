import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_Train_Test_Data():
    CSV_COLUMN_NAMES = ['Gaze left X', 'Gaze left Y', 'Gaze left Z',
                        'Gaze right X', 'Gaze right Y', 'Gaze right Z',
                        'Pupil left', 'Pupil right', 'HR', 'GSR', 'ConationLevel']

    CSV_COLUMN_NAMES_TEST = ['Gaze left X', 'Gaze left Y', 'Gaze left Z',
                             'Gaze right X', 'Gaze right Y', 'Gaze right Z',
                             'Pupil left', 'Pupil right', 'HR', 'GSR', 'ConationLevel',
                             'PredictedConation', 'GameState', 'TimeSinceStart']

    train_path = "TrainData.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    #train_feature = train.drop(['ConationLevel'], axis=1)
    train_feature = train
    train_feature = train_feature.iloc[0:-19]
    train_label = train.pop('ConationLevel')
    train_label = train_label.replace([1, 2, 3, 4], 0)
    train_label = train_label.replace([5, 6, 7], 1)
    train_label = train_label.iloc[0:-19]
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

#test_feature.hist(bins = 'auto')
#train_feature.plot(kind='density', subplots=True, sharex=False, layout=(4,4))

g = sns.PairGrid(train_feature)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6)


plt.tight_layout()
plt.show()
