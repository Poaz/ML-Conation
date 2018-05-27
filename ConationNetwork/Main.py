from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import pandas as pd
import argparse
import sklearn.model_selection as sk

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=10000, type=int,
                    help='number of training steps')

CSV_COLUMN_NAMES = ['Gaze3DpositionleftX', 'Gaze3DpositionleftY', 'Gaze3DpositionleftZ',
                    'Gaze3DpositionrightX', 'Gaze3DpositionrightY', 'Gaze3DpositionrightZ',
                    'Pupildiameterleft', 'Pupildiameterright', 'HR', 'GSR', 'ConationLevel']

def load_Train_Test_Data():
    CSV_COLUMN_NAMES = ['Gaze3DpositionleftX', 'Gaze3DpositionleftY', 'Gaze3DpositionleftZ',
                    'Gaze3DpositionrightX', 'Gaze3DpositionrightY', 'Gaze3DpositionrightZ',
                    'Pupildiameterleft', 'Pupildiameterright', 'HR', 'GSR', 'ConationLevel']

    CSV_COLUMN_NAMES_TEST = ['Gaze3DpositionleftX', 'Gaze3DpositionleftY', 'Gaze3DpositionleftZ',
                    'Gaze3DpositionrightX', 'Gaze3DpositionrightY', 'Gaze3DpositionrightZ',
                    'Pupildiameterleft', 'Pupildiameterright', 'HR', 'GSR', 'ConationLevel',
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



def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))
    dataset = dataset.shuffle(buffer_size=1000).repeat(count=None).batch(batch_size)

    return dataset.make_one_shot_iterator().get_next()


def eval_input_fn(features, labels=None, batch_size=None):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


def main(argv):
    args = parser.parse_args(argv[1:])

    # Call load_data() to parse the CSV file.
    (train_feature, train_label), (test_feature, test_label) = load_Train_Test_Data()

    #print(train_feature)

    my_feature_columns = []
    for key in train_feature.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    classifier = tf.estimator.DNNClassifier(feature_columns=my_feature_columns,
                                            hidden_units=[30, 30, 20, 10, 1],
                                            n_classes=2, activation_fn=tf.nn.sigmoid)

    classifier.train(input_fn=lambda: train_input_fn(train_feature, train_label, args.batch_size),
                     steps=args.train_steps)

    #Evaluate the model.
    eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_feature, test_label, args.batch_size))

    #print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
    print(eval_result)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
