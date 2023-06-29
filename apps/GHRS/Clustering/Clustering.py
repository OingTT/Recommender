import numpy as np
import pandas as pd

from warnings import warn
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, pairwise_distances

class Clustering():
  def __init__(self, CFG: dict) -> None:
    self.CFG = CFG

  def clustering(self, encoded_df: pd.DataFrame) -> np.ndarray:
    optimalKBySilhouetteScore = self.__findOptimalKBySilhouetteScore(encoded_df, 2, 10)
    optimalKByElbowMethod = self.__findOptimalKByElbowMethod(encoded_df, 2, 10)
    optimalKByGapStatistic = self.__findOptimalKByGapStatistic(encoded_df, 2, 10)
    if self.CFG.get('debug'):
      print(f'OptimalK by')
      print(f'Silhouette Score: {optimalKBySilhouetteScore}', end='\t')
      print(f'Elbow Method: {optimalKByElbowMethod}', end='\t')
      print(f'Gap Statistic: {optimalKByGapStatistic}')

    if self.CFG.get('optimal_k_method') == 'Silhouette':
      optimalK = optimalKBySilhouetteScore
    elif self.CFG.get('optimal_k_method') == 'Elbow':
      optimalK = optimalKByElbowMethod
    elif self.CFG.get('optimal_k_method') == 'Gap':
      optimalK = optimalKByGapStatistic

    clustering_result = self.__KMeans(df=encoded_df, K=optimalK, method='k-means++')
    # encoded_df['cluster_label'] = clustering_result.predict(encoded_df)
    return clustering_result.predict(encoded_df)

  def __call__(self, encoded_df: pd.DataFrame) -> None:
    warn("Clustering.__call__() is deprecated. Use Clustering.clustering() instead.", DeprecationWarning, stacklevel=2)
    return self.clustering(encoded_df)

  def __KMeans(self, df: pd.DataFrame, K: int=8, method: str='k-means++') -> KMeans:
    return KMeans(n_clusters=K, init=method, random_state=1, n_init='auto').fit(df)
  
  def __findOptimalKBySilhouetteScore(self, df: pd.DataFrame, minK: int=2, maxK: int=10) -> int:
    silhouette_scores = list()
    for k in range(minK, maxK, 1):
      kmeans = self.__KMeans(df=df, K=k, method='k-means++')
      cluster_labels = kmeans.predict(df)
      score = silhouette_score(df, cluster_labels)
      silhouette_scores.append(score)
    
    optimalK = silhouette_scores.index(max(silhouette_scores)) + minK
    return optimalK
  
  def __findOptimalKByElbowMethod(self, df: pd.DataFrame, minK: int=2, maxK: int=10) -> int:
    sse = list()
    for k in range(minK, maxK):
      kmeans = KMeans(n_clusters=k, init='k-means++', random_state=1, n_init='auto').fit(df)
      sse.append(kmeans.inertia_)

    optimalK = sse.index(min(sse)) + minK
    return optimalK      
  
  def __findOptimalKByGapStatistic(self, df: pd.DataFrame, minK: int=2, maxK: int=10, numIter: int=15) -> int:
    reference_scores = np.zeros(maxK - minK + 1)
    for k in range(minK, maxK + 1):
      cluster_scores = []
      for _ in range(numIter):
        kmeans = self.__KMeans(df=df, K=k, method='k-means++')
        cluster_scores.append(np.log(kmeans.inertia_))
      reference_scores[k - minK] = np.mean(cluster_scores)
    
    gaps: np.ndarray = reference_scores - np.log(np.mean(pairwise_distances(df)))
    optimalK = gaps.argmax() + minK
    return optimalK
    