import os

import itertools
import collections

import pandas as pd

from time import time
from tqdm import tqdm
from abc import *
from apps.utils.utils import save_pickle, load_pickle

def TimeTaken(func):
  def wrapper(*args, **kargs):
    startTime = time()
    print(func.__name__, 'begin at {}'.format(startTime))
    func(*args, **kargs)
    endTime = time()
    print(func.__name__, 'end at {}'.format(endTime))
    print(func.__name__, 'Taken {}/Sec'.format(endTime - startTime))
    return func
  return wrapper

class GraphFeature(metaclass=ABCMeta):
  preprocessed_data_dir = './preprocessed_data'
  
  # graphFeatures = list()
  # def addGraphFeatureWrapper(self, func):
  #   def wrapper(*args, **kargs):
  #     func(*args, **kargs)
  #     graphFeatureName = func.__name__.replace('_calc', '')
  #     self.graphFeatures.append(graphFeatureName)
  #     return func
  #   return wrapper

  def __init__(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, is_pred: bool=True):
    self.ratings = ratings_df
    self.users_df = users_df
    self.is_pred = is_pred

  def __call__(self, alpha_coef: float=0.005) -> pd.DataFrame:
    graphFeature_df_path = os.path.join(self.preprocessed_data_dir, 'graphFeature.pkl')
    self.graphFeature_df = load_pickle(graphFeature_df_path)
    self.graphFeature_df = None
    if not isinstance(self.graphFeature_df, pd.DataFrame):
      self._getGraph(alpha_coef=alpha_coef)
      self.graphFeature_df = self._getGraphFeatures()
      save_pickle(self.graphFeature_df, graphFeature_df_path)
    return self.graphFeature_df
  
  def _add_edge(self, edge: list) -> None:
    self.graph.add_edge(edge[0], edge[1])

  def _getGraph(self, alpha_coef):
    self._extendPairs()
    self._getEdgeList(alpha_coef)
    self._addGraphEdges() # UID가 String이라서, 그냥 넣으면 안됨

  def _extendPairs(self):
    self.pairs = list()
    grouped = self.ratings.groupby(['MID', 'Rating'])
    for key, group in tqdm(grouped, desc='_getGraph::extend'):
      for comb in itertools.combinations(group.index, 2):
        self.pairs.append(comb)

  def _getEdgeList(self, alpha_coef):
    counter = collections.Counter(self.pairs)
    alpha = alpha_coef * 3883  # param*i_no
    alpha = 0
    ### About 3~4 minute at aplha = 0.005 * 3883
    self.edge_list = map(list, collections.Counter(el for el in tqdm(counter.elements(), desc='_getGraph::map', total=132483307) if counter[el] >= alpha).keys())

  def graphFeature2DataFrame(self, col_name: str, graph_feature: pd.Series) -> None:
    self.users_df[col_name] = self.users_df.index.map(graph_feature)
    self.users_df[col_name] /= float(self.users_df[col_name].max())

  def _getGraphFeature(self) -> None:
    self._calcPagerank()
    self._calcDegreeCentrality()
    self._calcClosenessCentrality()
    self._calcBetweennessCentrality()
    self._calcLoadCentrality()
    self._calcAverageNeighborDegree()
    graphFeature_df = self.users_df[self.users_df.columns[0:]]
    graphFeature_df.fillna(0, inplace=True)
    self.graphFeature_df = graphFeature_df

  @abstractmethod
  def _getGraphFeatures(self) -> pd.DataFrame:
    ...
