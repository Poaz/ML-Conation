from __future__ import division
from pylab import plot, ylim, xlim, show, xlabel, ylabel, grid
import matplotlib.pyplot as plt
from numpy import linspace, loadtxt, ones, convolve
import numpy as numpy
import pandas as pd
import seaborn as sns


def load_data_one_set(label_name='ConationLevel'):
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel', 'PredictedConation'
                        ,'GameState', 'TimeSinceStart']

    train_path = r'Data90_9.txt'

    # Parse the local CSV file.
    data = pd.read_csv(filepath_or_buffer=train_path,
                        names=CSV_COLUMN_NAMES,
                        header=0, sep=',')

    dataset_features = data

    train_path_pred = r'Predictions_P90.csv'
    dataset_labels = pd.read_csv(filepath_or_buffer=train_path_pred,
                        names=['Predicted'],
                       header=0, sep=',')
    return (dataset_features, dataset_labels)

def movingaverage(interval, window_size):
    window= numpy.ones(int(window_size))/float(window_size)
    return numpy.convolve(interval, window, 'same')

(dataFeature, dataLabels)= load_data_one_set()

y = dataLabels['Predicted']
x = dataFeature['TimeSinceStart']


#plot(x,y,"k.")
y_av = movingaverage(y, 4000)


fig, ax = plt.subplots()
sns.regplot(x, y_av,"r", ax=ax)

xlabel("Time Since Start")
ylabel("Conation Level")
grid(True)

ax2 = ax.twinx()

ConationLevels = pd.read_csv(filepath_or_buffer=r'Data90_9.txt',
                             header=0, sep=',')
lm = sns.lmplot(y='ConationLevel', x='TimeSinceStart', hue="GameState", fit_reg=False, scatter_kws={"s": 5},
           data=ConationLevels, line_kws={"s": 1}, aspect=2, ax=ax2)

sns.plot.show()

