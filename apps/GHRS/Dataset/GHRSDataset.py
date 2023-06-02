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
  Training & Validation Step  : MovieLens + DB
  Predction Step              : Only DB
  '''
  # __occupation_dummy_prefix
  __GENDER_DUMMY_PREFIX: list = ['Gender_F', 'Gender_M']
  __AGE_DUMMY_PREFIX: list = [i for i in range(0, 6)]
  __OCCUPATION_DUMMY_PREFIX: list = ['Occupation_{}'.format(i) for i in range(11)]

  def __init__(
      self,
      CFG: dict,
      movieLensDir: str = './ml-1m',
      DataBaseLoader: DataBaseLoader=None,
    ) -> None:
    super(GHRSDataset, self).__init__()
    self.CFG = CFG
    self.movieLensDir = movieLensDir
    self.dataBaseLoader = DataBaseLoader
    self.__prepare_data()

  def __len__(self):
    '''
    Return dim of datasets 'features' (exclude UID)
    AutoEncoder객체 생성할 때는 GraphFeature들이 구해지지 않은 상태라 문제
    '''
    return len(list(self.GraphFeature_df.columns)) - 1 # exclude UID
  
  def __get_movie_lens(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users_df = pd.read_csv(
      os.path.join(self.movieLensDir, 'users.dat'),
      sep='\::',
      engine='python',
      names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
      dtype={
        'UID': 'str',
        'Gender': 'str',
        'Age': 'uint8',
        'Occupation': 'uint8',
        'Zip': 'string'
      }
    )
    ratings_df = pd.read_csv(
      os.path.join(self.movieLensDir, 'ratings.dat'),
      sep='\::',
      engine='python',
      names=['UID', 'CID', 'Rating', 'Timestamp'],
      dtype={
        'UID': 'str',
        'CID': 'uint16',
        'Rating': 'uint8',
        'Timestamp': 'uint64'
      }
    )
    ratings_df['ContentType'] = 'MOVIELENS'
    return users_df, ratings_df
  
  def __get_db_data(self, contentType: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    contentType: 'MOVIE' or 'TV' or 'ALL'
    '''
    __available_content_type = ['MOVIE', 'TV', 'ALL']
    assert self.dataBaseLoader is not None, 'GHRSDataset.dataBaseLoader must be set'
    if contentType not in __available_content_type:
      raise ValueError('ContentType Error')
    db_users = self.dataBaseLoader.getAllUsers()
    if contentType == 'ALL':
      db_ratings = self.dataBaseLoader.getAllReviews()
    else:
      db_ratings = self.dataBaseLoader.getReviewsByContentType(contentType)
    return db_users, db_ratings
  
  def __sample_movie_lens(self,
                          users_df: pd.DataFrame,
                          ratings_df: pd.DataFrame,
                          sample_rate: float=0.3,
                          random_state: int=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_len = int(len(users_df) * sample_rate)
    sampled_users_df = users_df.sample(sample_len, random_state=random_state)
    sampled_ratings_df = ratings_df[ratings_df['UID'].isin(sampled_users_df['UID'])]
    return sampled_users_df, sampled_ratings_df
  
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
    
  def __preprocess_users_df(self, users_df: pd.DataFrame) -> pd.DataFrame:
    for occupation in OCCUPATION_MAP.items(): # Apply occupation reduction
      users_df['Occupation'] = users_df['Occupation'].replace(occupation[0], occupation[1])
    users_df = self.__convert2Categorical(users_df, 'Occupation')
    users_df = self.__convert2Categorical(users_df, 'Gender')
    age_bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ['0', '1', '2', '3', '4', '5']
    users_df['bin'] = pd.cut(users_df['Age'], age_bins, labels=labels)
    users_df['Age'] = users_df['bin']
    users_df = self.__convert2Categorical(users_df, 'Age')
    users_df = users_df.drop(columns='bin')
    users_df = users_df.drop(columns='Zip')
    return users_df
  
  def __getTensorDataset(self, graphFeature: pd.DataFrame) -> TensorDataset:
    whole_x = torch.Tensor(np.array(graphFeature.values[:, 1:], dtype=np.float32))
    whole_y = torch.Tensor(graphFeature.index.to_list())

    whole_dataset = TensorDataset(whole_x, whole_y)
    
    return whole_dataset
    
  def __prepare_data(self) -> None:
    # Get DB Data
    users_df, ratings_df = self.__get_db_data('ALL')
    
    ml_users_df, ml_ratings_df = self.__get_movie_lens()
    self.ml_users_df, self.ml_ratings_df = ml_users_df, ml_ratings_df
    if self.CFG['sample_rate'] != 0. or self.CFG['train_ae']: # Use only DB Data
      ml_users_df, ml_ratings_df = self.__sample_movie_lens(
        ml_users_df,
        ml_ratings_df,
        sample_rate=self.CFG['sample_rate'],
        random_state=1
      )
      users_df = pd.concat([users_df, ml_users_df], axis=0)
      ratings_df = pd.concat([ratings_df, ml_ratings_df], axis=0)

    users_df = users_df.reset_index()

    users_df = self.__preprocess_users_df(users_df=users_df)

    if 'index' in users_df.columns.to_list():
      users_df = users_df.drop(['index'], axis=1)

    self.users_df, self.ratings_df = users_df, ratings_df

    self.GraphFeature = GraphFeature(self.CFG, users_df, ratings_df)
    self.GraphFeature_df = self.GraphFeature()

    whole_dataset = self.__getTensorDataset(self.GraphFeature_df)
    self.whole_dataset = whole_dataset
  
    train_set, valid_set = random_split(whole_dataset, [(1 - self.CFG['val_rate']), self.CFG['val_rate']])
    self.train_set, self.valid_set = train_set, valid_set

  def update_graph_feature(self, users_df: pd.DataFrame, ratings_df: pd.DataFrame) -> None:
    self.users_df, self.ratings_df = users_df, ratings_df
    self.GraphFeature = GraphFeature(users_df, ratings_df)
    self.GraphFeature_df = self.GraphFeature()
    whole_dataset = self.__getTensorDataset(self.GraphFeature_df)
    self.whole_dataset = whole_dataset
    train_set, valid_set = random_split(whole_dataset, [(1 - self.CFG['val_rate']), self.CFG['val_rate']])
    self.train_set, self.valid_set = train_set, valid_set

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
