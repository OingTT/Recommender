import os
import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from typing import Tuple
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from torch.utils.data import TensorDataset, DataLoader, random_split

from apps.apis.TMDB import TMDB
from .Const import OCCUPATION_MAP
from .DataBaseLoader import DataBaseLoader
from .GraphFeature.GraphFeature_GraphTool import GraphFeature_GraphTool as GraphFeature
# from .GraphFeature.GraphFeature_RAPIDS import GraphFeature_RAPIDS as GraphFeature

class GHRSDataset(pl.LightningDataModule):
  '''
  Training & Validation Step  : MovieLens + DB
  Predction Step              : Only DB
  '''
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
    Return dim of datasets 'features' (except UID)
    AutoEncoder객체 생성할 때는 GraphFeature들이 구해지지 않은 상태라 문제
    '''
    # return len(self.GraphFeature_df.columns) - 1 # except UID
    return 21

  def __convert2Categorical(self, df_X: pd.DataFrame, _X: str) -> pd.DataFrame:
    df_X = pd.get_dummies(df_X, columns=[_X], dummy_na=True)
    df_X = df_X.drop(columns=f'{_X}_nan')
    return df_X

  def __preprocess_users_df(self, users_df: pd.DataFrame) -> None:
    for occupation in OCCUPATION_MAP.items(): # Apply occupation reduction
      users_df['Occupation'] = users_df['Occupation'].replace(occupation[0], occupation[1])
    users_df = self.__convert2Categorical(users_df, 'Occupation')
    users_df = self.__convert2Categorical(users_df, 'Gender')
    age_bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ['1', '2', '3', '4', '5', '6']
    users_df['bin'] = pd.cut(users_df['Age'], age_bins, labels=labels)
    users_df['Age'] = users_df['bin']
    users_df = users_df.drop(columns='bin')
    users_df = self.__convert2Categorical(users_df, 'Age')
    users_df = users_df.drop(columns='Zip')
    return users_df
  
  def __sample_movie_lens(self,
                          users_df: pd.DataFrame,
                          ratings_df: pd.DataFrame,
                          sample_ratio: float=0.3,
                          random_state: int=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_len = int(len(users_df) * sample_ratio)
    sampled_users_df = users_df.sample(sample_len, random_state=random_state)
    sampled_ratings_df = ratings_df[ratings_df['UID'].isin(sampled_users_df['UID'])]
    return sampled_users_df, sampled_ratings_df
  
  def __prepare_data(self) -> None:
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
      names=['UID', 'MID', 'Rating', 'Timestamp'],
      dtype={
        'UID': 'str',
        'MID': 'uint16',
        'Rating': 'uint8',
        'Timestamp': 'uint64'
      }
    )

    # DB Data
    db_users, db_ratings = self.__get_db_data()

    # Sampling MovieLens: While training autoencoder model use whole dataset which including MovieLens
    if self.CFG['sample_rate'] != 1.:
      users_df, ratings_df = self.__sample_movie_lens(
        users_df,
        ratings_df,
        sample_ratio=self.CFG['sample_rate'],
        random_state=1
      )
    
    # While prediction use only DB data
    if self.CFG['sample_rate'] == 0.: # Use only DB Data
      users_df = db_users
      ratings_df = db_ratings

    # Merge DB and MovieLens
    if self.CFG['sample_rate'] != 0:
      users_df = pd.concat([users_df, db_users], axis=0)
      ratings_df = pd.concat([ratings_df, db_ratings], axis=0)

    self.users_df = users_df
    self.ratings_df = ratings_df

    users_df = self.__preprocess_users_df(users_df=users_df)

    self.GraphFeature = GraphFeature(ratings_df, users_df)
    self.GraphFeature_df = self.GraphFeature()

    whole_dataset = self.__getTensorDataset(self.GraphFeature_df)
    self.whole_dataset = whole_dataset
  
    train_set, valid_set = random_split(whole_dataset, [(1 - self.CFG['val_rate']), self.CFG['val_rate']])
    self.train_set, self.valid_set = train_set, valid_set
  

  def __getTensorDataset(self, graphFeature: pd.DataFrame) -> TensorDataset:
    whole_x = torch.Tensor(np.array(graphFeature.values[:, 1:], dtype=np.float32))
    whole_y = torch.Tensor(graphFeature.index.to_list())

    whole_dataset = TensorDataset(whole_x, whole_y)
    
    return whole_dataset 

  def __get_db_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    assert self.dataBaseLoader is not None, 'GHRSDataset.dataBaseLoader must be set'
    db_users = self.dataBaseLoader.getAllUsers()
    db_ratings = self.dataBaseLoader.getAllReviews()
    return db_users, db_ratings

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
