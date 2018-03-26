from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM
import keras as keras
import pandas as pd
import sklearn.model_selection as sk
from keras import regularizers
import tensorflow as tf
from keras.utils.np_utils import to_categorical
import numpy as np

##############################################################################
dropout = 0.3
epochs = 50
batchSize = 128
validationSplit = 0.2
classes = 7
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, decay=1e-6)
##############################################################################


def load_data(label_name='ConationLevel'):

    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X','Gaze 3D position right Y','Gaze 3D position right Z',
                        'Pupil diameter left','Pupil diameter right', 'HR', 'GSR', 'ConationLevel']

    train_path = "CombinedDataVelocity.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    train_features, test_features = sk.train_test_split(train, test_size=0.30, random_state=42)
    train_label, test_label = sk.train_test_split(train.pop(label_name), test_size=0.30, random_state=42)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

CallBack  = keras.callbacks.TensorBoard(log_dir='./Logs', histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
                            write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)

model = Sequential()

#model.add(LSTM(128, input_dim=12))
model.add(Dense(30, input_dim=11))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(25))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(7))
model.add(Activation('softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

(train_feature, train_label), (test_feature, test_label) = load_data()

one_hot_labels = keras.utils.to_categorical(train_label, num_classes=classes)

model.fit(train_feature, train_label, epochs=epochs, batch_size=batchSize, validation_split=validationSplit, callbacks=[CallBack])

loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=128)
print(loss_and_metrics)