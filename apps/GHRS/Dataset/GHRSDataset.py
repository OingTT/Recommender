import os
import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from typing import Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader, random_split

from apps.apis.TMDB import TMDB
from apps.GHRS.Dataset.CONST import OCCUPATION_MAP
from apps.GHRS.Dataset.DataBaseLoader import DataBaseLoader
from apps.GHRS.GraphFeature.GraphFeature_GraphTool import GraphFeature_GraphTool as GraphFeature

class GHRSDataset(pl.LightningDataModule):
  '''
  Features: User informations
  Target: Ratings about user
  Training & Validation Step  : MovieLens + DB
  Predction Step              : Only DB
  '''
  __GENDER_DUMMY_PREFIX: list = ['Gender_F', 'Gender_M']
  __AGE_DUMMY_PREFIX: list = [i for i in range(0, 6)]
  __OCCUPATION_DUMMY_PREFIX: list = ['Occupation_{}'.format(i) for i in range(11)]

  def __init__(
      self,
      CFG: dict,
      graph_features: pd.DataFrame
    ) -> None:
    super(GHRSDataset, self).__init__()
    self.CFG = CFG
    self.graph_features = graph_features
    self.__prepare_data()

  def __len__(self):
    '''
    Return dim of datasets 'features' (exclude UID)
    '''
    len_ = len(list(self.graph_features.columns)) - 1
    len_ = len_ + len(self.__GENDER_DUMMY_PREFIX) + len(self.__AGE_DUMMY_PREFIX) + len(self.__OCCUPATION_DUMMY_PREFIX) - 3
    return len_
  
  def __convert2Categorical(self, df_X: pd.DataFrame, _X: str) -> pd.DataFrame:
    if _X == 'Occupation':
      PREFIX = self.__OCCUPATION_DUMMY_PREFIX
    elif _X == 'Gender':
      PREFIX = self.__GENDER_DUMMY_PREFIX
      df_X['Gender'] = df_X['Gender'].replace('F', '0')
      df_X['Gender'] = df_X['Gender'].replace('M', '1')
    elif _X == 'Age':
      PREFIX = self.__AGE_DUMMY_PREFIX
      
    values = np.array(df_X[_X])
    # integer encode
    label_encoder = LabelEncoder()
    label_encoder = label_encoder.fit([_ for _ in range(len(PREFIX))])
    integer_encoded = label_encoder.transform(values)
    # binary encode
    onehot_encoder = OneHotEncoder(categories=[[_ for _ in range(len(PREFIX))]], sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df_temp = pd.DataFrame(onehot_encoded, columns=PREFIX)
    df_X = df_X.drop(_X, axis=1)
    df_X = pd.concat([df_X, df_temp], axis=1)
    return df_X
    
  def __preprocess_graph_features(self, graph_features: pd.DataFrame) -> pd.DataFrame:
    for occupation in OCCUPATION_MAP.items(): # Apply occupation reduction
      graph_features['Occupation'] = graph_features['Occupation'].replace(occupation[0], occupation[1])
    graph_features = self.__convert2Categorical(graph_features, 'Occupation')
    graph_features = self.__convert2Categorical(graph_features, 'Gender')
    age_bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ['0', '1', '2', '3', '4', '5']
    graph_features['bin'] = pd.cut(graph_features['Age'], age_bins, labels=labels)
    graph_features['Age'] = graph_features['bin']
    graph_features = self.__convert2Categorical(graph_features, 'Age')
    graph_features = graph_features.drop(columns='bin')
    graph_features = graph_features.drop(columns='Zip')
    return graph_features
  
  def __getTensorDataset(self, graph_features: pd.DataFrame) -> TensorDataset:
    whole_x = torch.Tensor(np.array(graph_features.values[:, 1:], dtype=np.float32))
    whole_y = torch.Tensor(graph_features.index.to_list())

    whole_dataset = TensorDataset(whole_x, whole_y)
    
    return whole_dataset
    
  def __prepare_data(self) -> None:
    self.graph_features = self.__preprocess_graph_features(self.graph_features)

    self.whole_dataset = self.__getTensorDataset(self.graph_features)

    self.train_set, self.valid_set = random_split(self.whole_dataset, [(1 - self.CFG['val_rate']), self.CFG['val_rate']])
    
  def train_dataloader(self):
    '''
    MovieLens + DB Data splitted
    '''
    return DataLoader(self.train_set, batch_size=self.CFG['batch_size'], num_workers=self.CFG['num_workers'], shuffle=True)
  
  def val_dataloader(self):
    '''
    MovieLens + DB Data splitted
    '''
    return DataLoader(self.valid_set, batch_size=self.CFG['batch_size'], num_workers=self.CFG['num_workers'], shuffle=False)
  
  def predict_dataloader(self):
    '''
    DB Data only
    '''
    return DataLoader(self.whole_dataset, batch_size=self.CFG['batch_size'], num_workers=self.CFG['num_workers'], shuffle=False)
