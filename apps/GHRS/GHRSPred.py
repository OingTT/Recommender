import os
import json

import pandas as pd

from typing import List, Dict

from apps.utils.Singleton import Singleton
from apps.GHRS.Dataset.DataBaseLoader import DataBaseLoader

class GHRSPred(metaclass=Singleton):
  def __init__(self, CFG: Dict) -> None:
    self.CFG = CFG

    self.databaseLoader = DataBaseLoader()

  def __load_clustered(self, target_UID: str) -> pd.DataFrame:
    target_cluster_label = int(self.databaseLoader.getUserClusteredByUserId(id=target_UID)['label'].iloc[0])
    userClustered = self.databaseLoader.getUserClusteredByClusterLabel(cluster_label=target_cluster_label)
    userClustered = userClustered.rename(columns={'id': 'UID', 'label': 'cluster_label'})
    return userClustered

  def __get_target_cluster_uids(self, target_UID: str) -> List[str]:
    '''
    타겟 유저가 속해있는 클러스터에 있는 타 유저들의 UID를 반환
    '''
    clustered: pd.DataFrame = self.__load_clustered()

    grouped = clustered.groupby('cluster_label', as_index=False)

    # target_cluster => cluster's label that UID is in
    target_cluster = None
    for cluster_label, indices in grouped.groups.items():
      user_in_cluster: pd.Series = clustered['UID'][indices]
      user_in_cluster = user_in_cluster.values.tolist()
      if target_UID in user_in_cluster:
        target_cluster = cluster_label
        break
    if target_cluster is None:
      raise ValueError('UID is not in any cluster')

    # target_cluster_df => dataframe of target cluster
    target_cluster_df = clustered[clustered['cluster_label'] == target_cluster]

    # target_cluster_uids => uids of target cluster
    target_cluster_uids: List[str] = map(str, target_cluster_df['UID'].values.tolist())

    return target_cluster_uids

  def predict_content(self, target_UID: str, contentType: str='MOVIE', top_N: int=20):
    '''
    return Dataframe of mean rating about target user's cluster
    columns: ['ContentType', 'CID', 'Rating']
    '''
    contentType = contentType.upper()

    target_cluster_uids = self.__get_target_cluster_uids(target_UID=target_UID)

    ratings: pd.DataFrame = self.databaseLoader.getAllReviews()

    # target_cluster_rating => ratings about every contentType of target cluster
    target_cluster_rating = ratings[ratings['UID'].isin(target_cluster_uids)]

    # target_cluster_rating => ratings about specified contentType of target cluster
    target_cluster_rating = target_cluster_rating[target_cluster_rating['ContentType'] == contentType]

    # target_cluster_rating_mean => mean rating of each content in target cluster
    target_cluster_rating_mean = target_cluster_rating.groupby('CID', as_index=False)['Rating'].mean()

    # target_user_rating => ratings of target user
    target_user_rating = ratings[ratings['UID'] == target_UID]['CID'].dropna().values.tolist()

    # target_cluster_rating_mean => drop contents that target user already watched
    target_cluster_rating_mean = \
      target_cluster_rating_mean[~target_cluster_rating_mean['CID'].isin(target_user_rating)]
    
    # target_cluster_rating_mean => drop contents that only have one rating
    target_cluster_rating_mean = \
      target_cluster_rating_mean[target_cluster_rating_mean['CID'].isin(ratings['CID'].value_counts()[ratings['CID'].value_counts() > 1].index)]
    
    # target_cluster_rating_mean_rating => sort by mean rating
    target_cluster_rating_mean.sort_values(by='Rating', ascending=False, inplace=True)
    
    # target_cluster_rating_mean_rating => top 10 movies
    target_cluster_rating_mean = target_cluster_rating_mean.iloc[: top_N]

    return self.__content_prediction_to_json(target_cluster_rating_mean, contentType)

  def predict_ott_comb(self, target_UID: str, topN: int=3):
    # target_cluster_uids => uids of target cluster
    target_cluster_uids = self.__get_target_cluster_uids(target_UID=target_UID)

    ratings: pd.DataFrame = self.databaseLoader.getAllReviews()
    
    # subscription => subscription of all users
    subscription = self.databaseLoader.getAllUserSubscribe()

    # subscription => subscription of target cluster
    subscription = subscription[ratings['UID'].isin(target_cluster_uids)]

    combinations = dict()
    for row_idx in range(len(subscription['UID'].unique())):
      user: pd.DataFrame = subscription[subscription['UID'] == subscription['UID'].iloc[row_idx]]
      user_sub = user['Subscription'].values
      # if len(user_sub) == 1:
      #   continue
      COMBINATION = sorted(user_sub).__str__()
      if COMBINATION in combinations.keys():
        combinations[COMBINATION] += 1
      else:
        combinations[COMBINATION] = 1

    comb_cnt = {k: v for k, v in sorted(combinations.items(), key=lambda item: item[1], reverse=True)}
    comb = [json.loads(key) for key, value in comb_cnt.items()]
    
    return comb[: topN]
  
  def predict_ott(self, target_UID: str, topN: int=20) -> List[dict]:
    # target_cluster_uids => uids of target cluster
    target_cluster_uids = self.__get_target_cluster_uids(UID=target_UID)

    # subscription => subscription of target cluster
    subscription = self.databaseLoader.getAllUserSubscribe()

    # subscription => subscription of target cluster
    subscription = subscription[subscription['UID'].isin(target_cluster_uids)]
    
    # subscription => group by OTT & count among UID(Count of how much users subscribing specific OTT)
    subscription = subscription.groupby('Subscription', as_index=False)['UID'].count()

    # subscription => sort by subscription count
    subscription.sort_values(by='UID', ascending=False, inplace=True)

    subscription = subscription.iloc[: topN]

    return self.__ott_prediction_to_json(subscription)
  
  def __content_prediction_to_json(self, rating_mean: pd.DataFrame, contentType: str) -> List[dict]:
    results = list()
    for i in range(len(rating_mean)):
      cid = rating_mean['CID'].iloc[i]
      rating = rating_mean['Rating'].iloc[i]
      results.append(dict(
        ContentType=contentType,
        ContentID=int(cid),
        Rating=int(rating)
      ))
    return results
  
  def __ott_prediction_to_json(self, subscription_count: pd.DataFrame) -> List[dict]:
    results = list()
    for i in range(len(subscription_count)):
      ott = subscription_count['Subscription'].iloc[i]
      count = subscription_count['UID'].iloc[i]
      results.append(dict(
        OTT=int(ott),
        Count=int(count)
      ))
    return results