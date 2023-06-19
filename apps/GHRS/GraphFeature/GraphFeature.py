import os
import itertools
import collections

import pandas as pd

from abc import *
from time import time
from tqdm import tqdm
from typing import List, Union, SupportsInt, SupportsFloat

import graph_tool as gt
from graph_tool import draw

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
  def __init__(self, CFG: dict):
    self.CFG = CFG

    self.GRAPH_FEATURE = os.path.join(CFG['preprocessed_data_dir'], 'GraphFeatures.pkl')

    self.graph_feature_calculators = self.__get_graph_feature_calculators

  def __call__(self, users: pd.DataFrame, ratings: pd.DataFrame, alpha_coef: float=0.005) -> pd.DataFrame:
    self.users = users
    self.ratings = ratings
    
    self._getGraph(alpha_coef=alpha_coef)
    self.graphFeature_df = self._getGraphFeatures()
    self.graphFeature_df = self.graphFeature_df.reset_index()

    if 'index' in self.graphFeature_df.columns.to_list():
      self.graphFeature_df = self.graphFeature_df.drop(['index'], axis=1)

    return self.graphFeature_df

  def _getGraph(self, alpha_coef):
    pairs = self._extendPairs()
    edge_list = self._getEdgeList(pairs, alpha_coef)
    self._addGraphEdges(edge_list)

  def _extendPairs(self) -> List:
    grouped = self.ratings.groupby(['CID', 'Rating'])
    print(grouped.groups)
    pairs = list()
    for _, group in tqdm(grouped, desc='_getGraph::extend'):
      for comb in itertools.combinations(group.index, 2):
        pairs.append(comb)
    return pairs

  def _getEdgeList(self, pairs: list, alpha_coef: Union[SupportsInt, SupportsFloat]) -> map:
    counter = collections.Counter(pairs)
    alpha = alpha_coef * 38
    edge_list = map(list, collections.Counter(el for el in tqdm(counter.elements(), desc='_getGraph::map') if counter[el] >= alpha).keys())
    return edge_list

  @abstractmethod
  def _addGraphEdges(self, edge_list: list) -> None:
    ...

  def concatGraphFeatureToUsers(self, col_name: str, graph_feature: pd.Series) -> pd.DataFrame:
    self.users[col_name] = self.users.index.map(graph_feature)
    self.users[col_name] /= float(self.users[col_name].max())

  def __get_graph_feature_calculators(self) -> List:
    calculator_list = list()
    for func_name in dir(self):
      if func_name.startswith('_calc'):
        calculator_list.append(func_name)
    return calculator_list

  def _getGraphFeatures(self) -> pd.DataFrame:
    self.graph_feature_values = dict()
    for func_name in dir(self):
      if func_name.startswith('_calc'):
        calculator = self.__getattribute__(func_name)
        gf = calculator()
        gf_name = func_name.replace('_calc', '')
        self.graph_feature_values[gf_name] = gf
    
    graph_features = self.users[self.users.columns[0: ]]
    graph_features = graph_features.fillna(0)
    print('GraphFeatures\n', graph_features)
    return graph_features

