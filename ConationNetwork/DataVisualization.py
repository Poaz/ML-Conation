import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time


def plot(data_file, prediction_file):
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel',
                        'PredictedConation', 'GameState', 'TimeSinceStart']

    DataPath = data_file

    # Parse the local CSV file.
    data = pd.read_csv(filepath_or_buffer=DataPath,
                       names=CSV_COLUMN_NAMES,
                       header=0, sep=',')

    DataPath2 = prediction_file

    # Parse the local CSV file.
    predictions = pd.read_csv(filepath_or_buffer=DataPath2,
                       header=0, sep=',')

    predictions.replace(0, 2.5)
    predictions.replace(1, 3.5)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sns.lmplot(y="ConationLevel", x="TimeSinceStart", hue="GameState", fit_reg=False, data=data, scatter_kws={"s": 10})

    sns.regplot(y=predictions['0'], x=data["TimeSinceStart"], fit_reg=False, scatter_kws={"s": 5})


    plt.axhline(y=3, ls=":", c=".5")
    ax.set_xlabel('Time Since Start')
    ax.set_ylabel('Conation Level')

    plt.show()



def corr_plots():
    CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                        'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                        'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel',
                        'PredictedConation', 'GameState',
                        'TimeSinceStart']

    DataPath = "Data01_8.txt"

    # Parse the local CSV file.
    data = pd.read_csv(filepath_or_buffer=DataPath,
                       names=CSV_COLUMN_NAMES,
                       header=0, sep=',')
    cor = data.corr() #Calculate the correlation of the above variables
    plot = sns.heatmap(cor) #Plot the correlation as heat map

    plt.show()

    def doKmeans(X, nclust=2):
        model = KMeans(nclust)
        model.fit(X)
        clust_labels = model.predict(X)
        cent = model.cluster_centers_
        return (clust_labels, cent)

    clust_labels, cent = doKmeans(data[['ConationLevel']], 2)
    kmeans = pd.DataFrame(clust_labels)
    data[['ConationLevel']].insert((data[['ConationLevel']].shape[1]),'kmeans',kmeans)

    fig = plt.figure()
    ax = fig.add_subplot(222)
    scatter = ax.scatter(data['Gaze 3D position left X'],data['Gaze 3D position left Y'],c=kmeans[0],s=10)
    ax.set_title('K-Means Clustering')
    ax.set_xlabel('GDP per Capita')
    ax.set_ylabel('Corruption')
    plt.colorbar(scatter)
    plt.show()

    colours = {0, 1, 2, 3, 4, 5, 6}
    pca = PCA(2)
    projected = pca.fit_transform(data)
    scatter2 = plt.scatter(projected[:, 0], projected[:, 1])
    #plt.colorbar(scatter2);
    plt.show()
