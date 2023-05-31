import os
import itertools
import collections

import pandas as pd

from abc import *
from time import time
from tqdm import tqdm
from apps.utils.utils import save_pickle, load_pickle

def TimeTaken(func):
  def wrapper(*args, **kargs):
    startTime = time()
    print(func.__name__, 'begin at {}'.format(startTime))
    func_res = func(*args, **kargs)
    endTime = time()
    print(func.__name__, 'end at {}'.format(endTime))
    print(func.__name__, 'Taken {}/Sec'.format(endTime - startTime))
    return func_res
  return wrapper

class GraphFeature(metaclass=ABCMeta):
  preprocessed_data_dir = './preprocessed_data'

  def __init__(self, CFG: dict, users_df: pd.DataFrame, ratings_df: pd.DataFrame):
    self.CFG = CFG
    self.users_df = users_df
    self.ratings = ratings_df

  def __call__(self, alpha_coef: float=0.005) -> pd.DataFrame:
    graphFeature_df_path = os.path.join(self.preprocessed_data_dir, 'graphFeature.pkl')
    
    if self.CFG['train_ae']:
      self.graphFeature_df = load_pickle(graphFeature_df_path)
    else:
      self.graphFeature_df = None
    if not isinstance(self.graphFeature_df, pd.DataFrame):
      self._getGraph(alpha_coef=alpha_coef)
      self.graphFeature_df = self._getGraphFeatures()
      self.graphFeature_df = self.graphFeature_df.reset_index()

      if 'index' in self.graphFeature_df.columns.to_list():
        self.graphFeature_df = self.graphFeature_df.drop(['index'], axis=1)

      if self.CFG['train_ae']:
        save_pickle(self.graphFeature_df, graphFeature_df_path)
    return self.graphFeature_df
  
  def _add_edge(self, edge: list) -> None:
    self.G.add_edge(edge[0], edge[1])

  def _getGraph(self, alpha_coef):
    self._extendPairs()
    self._getEdgeList(alpha_coef)
    self._addGraphEdges() # UID가 String이라서, 그냥 넣으면 안됨

  def _extendPairs(self):
    grouped = self.ratings.groupby(['CID', 'Rating'])
    # self.pairs = load_pickle('./pairs.pkl')
    # if isinstance(self.pairs, list):
    #   return
    self.pairs = list()
    for key, group in tqdm(grouped, desc='_getGraph::extend'):
      for comb in itertools.combinations(group.index, 2):
        self.pairs.append(comb)
    print('pairs.len: ', len(self.pairs))
    # save_pickle(self.pairs, './pairs.pkl')

  def _getEdgeList(self, alpha_coef):
    counter = collections.Counter(self.pairs)
    ### About 3~4 minute at aplha = 0.005 * 3883
    # alpha = alpha_coef * 3883  # param*i_no
    alpha = alpha_coef * 38
    self.edge_list = map(list, collections.Counter(el for el in tqdm(counter.elements(), desc='_getGraph::map', total=132483307) if counter[el] >= alpha).keys())

  def graphFeature2DataFrame(self, col_name: str, graph_feature: pd.Series) -> None:
    self.users_df[col_name] = self.users_df.index.map(graph_feature)
    self.users_df[col_name] /= float(self.users_df[col_name].max())

  @abstractmethod
  def _getGraphFeatures(self) -> pd.DataFrame:
    ...

  @abstractmethod
  def _addGraphEdges(self) -> None:
    ...