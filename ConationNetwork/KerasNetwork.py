from keras.layers import Dense, Dropout, Activation, LSTM
from keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Dense
import keras as keras
import pandas as pd
import sklearn.model_selection as sk
import tensorflow as tf
from tensorflow.python.tools import freeze_graph
from keras import backend as K


##############################################################################
dropout = 0.3
epochs = 50
batchSize = 128
validationSplit = 0.15
classes = 7
sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
adam = Adam(lr=0.001, decay=1e-6)
model_path = ''
env_name = 'ConationModel'
##############################################################################
"""
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

from keras import backend as K
K.tensorflow_backend._get_available_gpus()
"""

def load_data(label_name='ConationLevel'):

    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel']

    #CSV_COLUMN_NAMES = ['angleleft', 'angleright',
    #                    'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel']

    train_path = "CombinedDataNoZerosAbsVelocityOnEyes.csv"

    # Parse the local CSV file.
    train = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    train_features, test_features = sk.train_test_split(train, test_size=0.30, random_state=42)
    train_features = train_features.drop(['ConationLevel'], axis=1)
    test_features = test_features.drop(['ConationLevel'], axis=1)

    train_label, test_label = sk.train_test_split(train.pop(label_name), test_size=0.30, random_state=42)

    # Return four DataFrames.
    return (train_features, train_label), (test_features, test_label)


def save_model(sess, saver, model_path=""):
    """
    Saves current model to checkpoint folder.
    :param sess: Current Tensorflow session.
    :param model_path: Designated model path.
    :param steps: Current number of steps in training process.
    :param saver: Tensorflow saver for session.
    """
    last_checkpoint = model_path + '/model.cptk'
    saver.save(sess=sess, save_path=last_checkpoint)
    tf.train.write_graph(sess.graph_def, model_path, 'raw_graph_def.pb', as_text=False)
    print("Saved Model")


def export_graph(model_path, env_name="env", target_nodes="action"):
    """
    Exports latest saved model to .bytes format for Unity embedding.
    :param model_path: path of model checkpoints.
    :param env_name: Name of associated Learning Environment.
    :param target_nodes: Comma separated string of needed output nodes for embedded graph.
    """
    ckpt = tf.train.get_checkpoint_state(model_path)
    freeze_graph.freeze_graph(input_graph=model_path + '/raw_graph_def.pb',
                              input_binary=True,
                              input_checkpoint=ckpt.model_checkpoint_path,
                              output_node_names=target_nodes,
                              output_graph=model_path + '/' + env_name + '.bytes',
                              clear_devices=True, initializer_nodes="", input_saver="",
                              restore_op_name="save/restore_all", filename_tensor_name="save/Const:0")


def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
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

#with tf.Session() as sess:
#    sess.run(tf.global_variables_initializer())
#saver = tf.train.Saver()
model = Sequential()

#model.input(input_dim=10, name="input")

model.add(Dense(40, input_dim=10, name="dense_one"))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(40, name="dense_two"))
model.add(Activation('relu'))
model.add(Dropout(dropout))
model.add(Dense(7, activation='softmax'))
model.summary()
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

(train_feature, train_label), (test_feature, test_label) = load_data()

one_hot_labels = keras.utils.to_categorical(train_label, num_classes=classes)

#, callbacks=[CallBack]
model.fit(train_feature, train_label, epochs=epochs, batch_size=batchSize)

loss_and_metrics = model.evaluate(test_feature, test_label, batch_size=128)
print(loss_and_metrics)

model.save('ConationModel.HDF5')

# Create, compile and train model...
#frozen_graph = freeze_session(K.get_session(), output_names=[out.op.name for out in model.outputs])
#tf.train.write_graph(frozen_graph, model_path, "my_model.pb", as_text=False)
