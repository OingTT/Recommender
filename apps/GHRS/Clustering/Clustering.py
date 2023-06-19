import pandas as pd

from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

class Clustering():
    def __init__(self):
      ...

    def __call__(self, encoded_df: pd.DataFrame) -> None:
      optimalK = self.__findOptimalKBySilhouetteScore(encoded_df, 2, 10)
      print(f"Clustering with K={optimalK}")
      clustering_result = self.KMeans(df=encoded_df, K=optimalK, method='k-means++')
      encoded_df['cluster_label'] = clustering_result
      return encoded_df

    def KMeans(self, df: pd.DataFrame, K: int=8, method: str='k-means++') -> None:
      return KMeans(n_clusters=K, init=method, random_state=1, n_init='auto').fit_predict(df)
    
    def __findOptimalKBySilhouetteScore(self, df: pd.DataFrame, minK: int=2, maxK: int=10) -> int:
      bestScore = -1
      bestK = -1
      for k in range(minK, maxK, 1):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1, n_init='auto').fit_predict(df)
        score = silhouette_score(df, kmeans)
        if bestScore < score:
          bestScore = score
          bestK = k
      return bestK
    
    def __findOptimalKByElbowMethod(self, df: pd.DataFrame, minK: int=2, maxK: int=10) -> int:
      wcss = list()
      for k in range(minK, maxK):
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1, n_init='auto').fit(df)
        wcss.append(kmeans.inertia_)
      diffs = []
      for i in range(len(wcss) - 1):
          diff = wcss[i] - wcss[i+1]
          diffs.append(diff)
      elbow_index = diffs.index(max(diffs)) + 1
      bestK = elbow_index + 1
      return bestK
