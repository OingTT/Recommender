import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

class Cluster():
    def __init__(self, encoded_df: pd.DataFrame) -> None:
      self._encoded_df = encoded_df
      self._cluster = None
      self._silhouette_score = None

    def KMeans(self, K: int=8, method: str='k-means++') -> None:
      return KMeans(n_clusters=K, init=method).fit_predict(self._encoded_df)
    
    def _findOptimalKWithSilhouetteScore(self, minK: int=2, maxK: int=10) -> dict():
      bestScore = -1
      for k in range(minK, maxK, 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1, n_init='auto').fit(self._encoded_df)
        score = silhouette_score(self._encoded_df, kmeans.labels_)
        if score > bestScore:
          bestScore = score
          bestK = k
      return bestK
    
    def _findOptimalKWithElbowMethod(self, minK: int=2, maxK: int=10) -> dict():
      wcss = list()
      for k in range(minK, maxK):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1, n_init='auto').fit(self._encoded_df)
        wcss.append(kmeans.inertia_)
      diffs = []
      for i in range(len(wcss) - 1):
          diff = wcss[i] - wcss[i+1]
          diffs.append(diff)
      elbow_index = diffs.index(max(diffs)) + 1
      bestK = elbow_index + 1
      return bestK

    def DBSCAN(self, eps: float=0.3, min_samples: int=10) -> None:
      return DBSCAN(eps=eps, min_samples=min_samples).fit_predict(self._encoded_df)


# from sklearn.datasets import make_blobs
# import matplotlib.pyplot as plt

# X, y = make_blobs(n_samples=1000, centers=4, n_features=2, random_state=1)
# df = pd.DataFrame(X, columns=['x', 'y'])
# cluster = Cluster(df)
# # df['label_dbscan'] = cluster.DBSCAN(eps=0.5, min_samples=10)


# optimalK = cluster._findOptimalKWithSilhouetteScore(2, 30)
# s_ = f'label_kmeans_S: {optimalK}'
# df[s_] = cluster.KMeans(K=optimalK, method='k-means++')
# optimalK = cluster._findOptimalKWithElbowMethod(2, 30)
# e_ = f'label_kmeans_E: {optimalK}'
# df[e_] = cluster.KMeans(K=optimalK, method='k-means++')

# fig, axs = plt.subplots(1, 2)

# df.plot.scatter(x='x', y='y', c=s_, colormap='viridis', ax=axs[0])
# df.plot.scatter(x='x', y='y', c=e_, colormap='viridis', ax=axs[1])
# plt.show()