import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import time



CSV_COLUMN_NAMES = ['Gaze 3D position left X', 'Gaze 3D position left Y', 'Gaze 3D position left Z',
                    'Gaze 3D position right X', 'Gaze 3D position right Y', 'Gaze 3D position right Z',
                    'Pupil diameter left', 'Pupil diameter right', 'HR', 'GSR', 'ConationLevel']

DataPath = "Binary - Kopi.csv"
DataPath2 = "Data01.txt"

# Parse the local CSV file.
data = pd.read_csv(filepath_or_buffer=DataPath,
                   names=CSV_COLUMN_NAMES,
                   header=0, sep=',')
data2 = pd.read_csv(filepath_or_buffer=DataPath2,
                   names=CSV_COLUMN_NAMES,
                   header=0, sep=',')


sns.regplot(y=data["ConationLevel"], x=data.index.values, fit_reg=False)
plt.show()
#sns.regplot(y=data2["GDLX"], x=data2.index.values, fit_reg=False)
#plt.show()






def corr_plots():
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
