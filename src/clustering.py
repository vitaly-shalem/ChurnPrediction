import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def normalize_clustering_data(df):
    """ Normalize data by calculating z-score """
    df_norm = (df-df.mean())/df.std()
    return df_norm


def scale_clustering_data(df):
    """ Scale data using Standard scaler """
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df) 
    return df_scaled


def train_kmeans(df, num_clusters, scale=False, zscore=False):
    # Since this is a distance based algorithm it is often a good idea to scale our features 
    if scale:
       df = scale_clustering_data(df) 
    elif zscore:
       df = normalize_clustering_data(df)
    
    kmeans = KMeans(
        init="random",
        n_clusters=num_clusters,
        n_init='auto',
        max_iter=300,
        random_state=0
    )
    kmeans.fit(df)

    return  kmeans


def plot_elbow(df_data, bScale, bZscore):
    """ xxx """
    wcss = {}
    for i in range(1, 11):
        test_trained_model = train_kmeans(df_data, i, scale=bScale, zscore=bZscore)
        wcss[i] = test_trained_model.inertia_
    plt.plot(wcss.keys(), wcss.values(), 'gs-')
    plt.xlabel("Values of 'k'")
    plt.ylabel('WCSS')
    plt.show()


def apply_pca(df, model):
    """ xxx """
    pca = PCA(n_components=2)
    components = pca.fit_transform(df)
    data_pca_transformed = pd.DataFrame(data=components, columns=['PCA1', 'PCA2'])
    centers=pca.transform(model.cluster_centers_)
    return data_pca_transformed, centers


def plot_clusters(df_pca, cntrs, model):
    """ xxx """
    plt.scatter(df_pca['PCA1'], df_pca['PCA2'], c=model.labels_)
    plt.scatter(cntrs[:,0], cntrs[:,1], marker='x', s=100, c='red')
    plt.xlabel('PCA1')
    plt.ylabel('PCA2')
    plt.title('Customer Clusters')
    plt.tight_layout()
    plt.show()
