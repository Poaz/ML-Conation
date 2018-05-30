from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import itertools
import keras.backend as k

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
class_names = ['Low', 'High']
##############################################################################


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
                print(self.currentStep)
                if((features.shape[0]-1000) == self.currentStep):
                    self.currentStep = 0
            yield batch_features, batch_labels


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


model = Sequential()

model.add(LSTM(32, return_sequences=True, input_shape=(200, 10)))
model.add(LSTM(32, return_sequences=True))
model.add(LSTM(32))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['binary_accuracy'])

(train_feature, train_label), (test_feature, test_label) = load_Train_Test_Data()

model.fit_generator(MainGenerator(train_feature, train_label, 200).generator(), steps_per_epoch=(train_feature.shape[0]/210)-1, epochs=2)

#model.fit_generator(generator(train_feature, train_label, 200), steps_per_epoch=200, epochs=1)

loss_and_metrics = model.evaluate_generator(MainGenerator(test_feature, test_label, 200).generator(), steps=(test_feature.shape[0]/210)-1)

print("\n" + "Loss: " + str(loss_and_metrics[0]) + "\n" + "Accuracy: " + str(loss_and_metrics[1] * 100) + "%")

model.save('ConationModel_Stacked_LSTM.HDF5')


"""
# Compute confusion matrix
cnf_matrix = confusion_matrix(test_label, loss_and_metrics)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()
"""

