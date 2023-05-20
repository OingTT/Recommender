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

class GHRSDataset(pl.LightningDataModule):
  '''
  TODO X: Features, Y: UID ?
  '''
  def __init__(
      self,
      CFG: dict,
      movieLensDir: str = './ml-1m',
      DataBaseLoader: DataBaseLoader=None,
    ) -> None:
    super(GHRSDataset, self).__init__()
    self.movieLensDir = movieLensDir
    self.CFG = CFG
    # self.batch_size = self.
    # self.num_workers = num_workers
    # self.val_rate = val_rate
    # self.is_pred = is_pred
    self.dataBaseLoader = DataBaseLoader
    self.tmdb_api = TMDB()
    self.__prepare_data()

  def __len__(self):
    '''
    Return dim of datasets 'feature' (except UID)
    '''
    return self.whole_data.shape[1] - 1 # except UID

  def __convert2Categorical(self, df_X: pd.DataFrame, _X: str) -> pd.DataFrame:
    values = np.array(df_X[_X])
    # Encode to integer
    labelEncoder = LabelEncoder()
    integer_encoded = labelEncoder.fit_transform(values)
    # Encode to binary
    onehot_encoder = OneHotEncoder(sparse_output=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    df_X = df_X.drop(columns=_X)
    for j in range(integer_encoded.max() + 1):
      df_X.insert(loc=j + 1, column=str(_X) + str(j + 1), value=onehot_encoded[:, j])
    return df_X

  def __preprocess_users_df(self, users_df: pd.DataFrame) -> None:
    for occupation in OCCUPATION_MAP.items(): # Apply occupation reduction
      users_df['Occupation'] = users_df['Occupation'].replace(occupation[1], occupation[0])
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
  
  def __prepare_data(self) -> None:
    users_df = pd.read_csv(
      os.path.join(self.movieLensDir, 'users.dat'),
      sep='\::',
      engine='python',
      names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
      dtype={
        'UID': 'string',
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
        'UID': 'string',
        'MID': 'uint16',
        'Rating': 'uint8',
        'Timestamp': 'uint64'
      }
    )
    # IMDB-ID to TMDB-ID is take too long so I will use TMDB-ID temporarily
    # ratings_df['MID'] = self.tmdb_api.get_tmdb_ids(ratings_df['MID'].values)
    db_users, db_ratings = self.__get_db_data(ratings_df=ratings_df, users_df=users_df)
    
    # Merge DB and MovieLens
    users_df = pd.concat([users_df, db_users], axis=0)
    ratings_df = pd.concat([ratings_df, db_ratings], axis=0)

    # For test
    users_df = db_users
    ratings_df = db_ratings

    users_df = self.__preprocess_users_df(users_df=users_df)

    self.GraphFeature = GraphFeature(ratings_df, users_df)
    self.GraphFeature_df = self.GraphFeature()

    whole_x = torch.Tensor(self.GraphFeature_df.values[:, 1:])
    whole_y = torch.Tensor(self.GraphFeature_df.values[:, 0])
    
    self.whole_dataset = TensorDataset(whole_x, whole_y)

    self.train_set, self.valid_set = random_split(self.whole_dataset, [(1 - self.val_rate), self.val_rate])
  
  def __get_db_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Movie Lens Dataset을 사용하는 이유가 Cold start problem 해결하기 위함이므로
    데이터가 충분히 많은 경우에는 Movie Lens Dataset을 사용하지 않고
    DataBaseLoader만을 사용해 데이터를 불러올 예정.
    '''
    assert self.dataBaseLoader is not None, 'GHRSDataset.dataBaseLoader must be set'
    db_users = self.dataBaseLoader.getAllUsers()
    db_ratings = self.dataBaseLoader.getAllReviews()
    return db_users, db_ratings

  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=self.CFG['batch-size'], num_workers=self.CFG['num-workers'], shuffle=True)
  
  def val_dataloader(self):
    return DataLoader(self.valid_set, batch_size=self.CFG['batch-size'], num_workers=self.CFG['num-workers'], shuffle=False)
  
  def predict_dataloader(self):
    return DataLoader(self.whole_dataset, batch_size=self.CFG['batch-size'], num_workers=self.CFG['num-workers'], shuffle=False)
