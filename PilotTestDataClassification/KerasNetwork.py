from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.models import Sequential
from keras.layers import Dense
import keras as keras
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np

def load_data(label_name='EyeMovementType'):

    CSV_COLUMN_NAMES = ['GazePointX', 'GazePointY',
                        'GazeDirectionLeftX', 'GazeDirectionLeftY', 'GazeDirectionLeftZ',
                        'GazeDirectionRightX', 'GazeDirectionRightY', 'GazeDirectionRightZ',
                        'GazeEventDuration', 'FixationPointX', 'FixationPointY', 'EyeMovementType']

    train_path = "Data.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    train_features, test_features = sk.train_test_split(train, test_size=0.33, random_state=42)
    train_label, test_label = sk.train_test_split(train.pop(label_name), test_size=0.33, random_state=42)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

model = Sequential()

model.add(Dense(64, activation='relu', input_dim=12))
#model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(4, activation='relu'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

(train_feature, train_label), (test_feature, test_label) = load_data()

#train_label = np.random.randint(10, size=(660321, 1))

one_hot_labels = keras.utils.to_categorical(train_label, num_classes=4)

model.fit(train_feature, train_label, epochs=20, batch_size=128)

loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=128)
print(loss_and_metrics)