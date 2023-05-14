import os

import pickle
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

  def __init__(self, ratings_df: pd.DataFrame, users_df: pd.DataFrame, is_pred: bool=True):
    self.ratings = ratings_df
    self.users_df = users_df
    self.is_pred = is_pred

  def __call__(self, alpha_coef: float=0.005) -> pd.DataFrame:
    graphFeature_df_path = os.path.join(self.preprocessed_data_dir, 'graphFeature.pkl')
    self.graphFeature_df = load_pickle(graphFeature_df_path)
    if not isinstance(self.graphFeature_df, pd.DataFrame):
      self._getGraph(alpha_coef=alpha_coef)
      self._getGraphFeature()
      save_pickle(self.graphFeature_df, graphFeature_df_path)
    return self.graphFeature_df

  def _getGraph(self, alpha_coef):
    self._extendPairs()
    self._getEdgeList(alpha_coef)
    self._addGraphEdges()

  def _extendPairs(self):
    # self.pairs = load_pickle('./pairs.pkl')
    self.pairs = False
    if not self.pairs and self.is_pred:
      self.pairs = list()
      grouped = self.ratings.groupby(['MID', 'Rating'])
      for key, group in tqdm(grouped, desc='_getGraph::extend'):
        for comb in itertools.combinations(group['UID'], 2):
          self.pairs.extend(list(comb))
      print(self.pairs[:10])
      # save_pickle(self.pairs, './pairs.pkl')

  def _getEdgeList(self, alpha_coef):
    counter = collections.Counter(self.pairs)
    alpha = alpha_coef * 3883  # param*i_no
    ### About 3~4 minute at aplha = 0.005 * 3883
    edge_list_path = os.path.join(self.preprocessed_data_dir, 'edge_list_{:.3f}.pkl'.format(alpha))
    self.edge_list = load_pickle(edge_list_path)
    if not self.edge_list and self.is_pred:
      self.edge_list = map(list, collections.Counter(el for el in tqdm(counter.elements(), desc='_getGraph::map', total=132483307) if counter[el] >= alpha).keys())
      save_pickle(self.edge_list, edge_list_path)

  def graphFeature2DataFrame(self, col_name: str, graph_feature: pd.Series) -> None:
    self.users_df[col_name] = self.users_df['UID'].map(graph_feature)
    self.users_df[col_name] /= float(self.users_df[col_name].max())

  def _getGraphFeature(self) -> None:
    self._calcPagerank()
    self._calcDegreeCentrality()
    self._calcClosenessCentrality()
    self._calcBetweennessCentrality()
    self._calcLoadCentrality()
    self._calcAverageNeighborDegree()
    graphFeature_df = self.users_df[self.users_df.columns[1:]]
    graphFeature_df.fillna(0, inplace=True)
    self.graphFeature_df = graphFeature_df

  @abstractmethod
  def _addGraphEdges(self) -> None:
    ...

  @abstractmethod
  def _calcPagerank(self) -> None:
    ...

  @abstractmethod
  def _calcDegreeCentrality(self) -> None:
    ...

  @abstractmethod
  def _calcClosenessCentrality(self) -> None:
    ...

  @abstractmethod
  def _calcBetweennessCentrality(self) -> None:
    ...

  @abstractmethod
  def _calcLoadCentrality(self) -> None:
    ...

  @abstractmethod
  def _calcAverageNeighborDegree(self) -> None:
    ...

