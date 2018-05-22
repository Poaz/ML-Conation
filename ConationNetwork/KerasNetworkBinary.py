from keras.layers import Dense, Dropout, Activation, LSTM, BatchNormalization, LeakyReLU
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
import keras as keras
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from keras import backend as K
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold


##############################################################################
dropout = 0.25
epochs = 50
batchSize = 128
validationSplit = 0.15
classes = 2
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, decay=1e-6)
model_path = ''
env_name = 'ConationModel'
##############################################################################

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
    #test_features = test_features.drop(['GSR'], axis=1)

    train_label, test_label = sk.train_test_split(train.pop(label_name), test_size=0.20, random_state=42)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)

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

def save_model(sess, saver, model_path=""):

    last_checkpoint = model_path + '/model.cptk'
    saver.save(sess=sess, save_path=last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action"):

    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):

    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        print(output_names)
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def,
                                                      output_names, freeze_var_names)
        return frozen_graph


CallBack = keras.callbacks.TensorBoard(log_dir='./Logs', histogram_freq=1, batch_size=32, write_graph=True,
                                        write_grads=False, write_images=False, embeddings_freq=0,
                                        embeddings_layer_names=None, embeddings_metadata=None)

def Keras_model():
    model = Sequential()
    model.add(Dense(30, input_dim=10, kernel_initializer='normal'))
    #model.add(BatchNormalization())
    model.add(Dropout(dropout))
    model.add(Dense(30, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(20, activation='sigmoid'))
    model.add(Dropout(dropout))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()
    #conation 1-4 = 0; 5-7 = 1
    model.compile(loss='binary_crossentropy',
              optimizer=adam,
              metrics=['binary_accuracy'])
    return model

(train_feature, train_label), (test_feature, test_label) = load_data()

#, callbacks=[CallBack]

#Train and evaluate model
model = Keras_model()
model.fit(train_feature, train_label, epochs=epochs, batch_size=batchSize)
loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=128)


#Print results
print("\n" + "Loss: " + str(loss_and_metrics[0]) + "\n" + "Accuracy: " + str(loss_and_metrics[1]*100) + "%")

#Save model
model.save('ConationModel.HDF5')

#frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, model_path, "my_model.pb", as_text=False)


"""
(train_feature, train_label) = load_data_one_set()
one_hot_labels = keras.utils.to_categorical(train_label, num_classes=classes)
estimator = KerasClassifier(build_fn=Keras_model(), epochs=epochs, batch_size=batchSize, verbose=0)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(estimator, train_feature, train_label, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
"""