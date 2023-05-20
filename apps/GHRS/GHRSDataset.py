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
  TODO Latent Vector만들고 UID랑 합쳐야 함 => 어떤 유저에 대한 Latent Vector인지 알아야 하기 떄문
  현재 생각으로는 Input data에 UID를 포함해서 Autoencoder에 입력 후 Autoencoder에서 UID를 제외하는 방법으로 구현
  Input data는 TensorDataset으로 들어가니 0번째 feature에 UID를 넣어서 Autoencoder에 입력 후
  0번째 index feature를 제외하고 Latent Vector를 추출 후 UID와 합치는 방법으로 구현
  '''
  def __init__(
      self,
      movieLensDir: str = './ml-1m',
      DataBaseLoader: DataBaseLoader=None,
      batch_size: int = 64,
      num_workers: int = 0,
      val_rate: float=0.2,
      is_pred: bool=False,
    ) -> None:
    super(GHRSDataset, self).__init__()
    self.movieLensDir = movieLensDir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.val_rate = val_rate
    self.dataBaseLoader = DataBaseLoader
    self.is_pred = is_pred
    self.tmdb_api = TMDB()
    self._prepare_data()

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

  def __preprocess(self) -> None:
    self.Users_df = self.__convert2Categorical(self.Users_df, 'Occupation')
    self.Users_df = self.__convert2Categorical(self.Users_df, 'Gender')
    # 연령대 수정
    age_bins = [0, 10, 20, 30, 40, 50, 100]
    labels = ['1', '2', '3', '4', '5', '6']
    assert len(age_bins) == len(labels) + 1, 'age_bins and labels must be same length'
    self.Users_df['bin'] = pd.cut(self.Users_df['Age'], age_bins, labels=labels)
    self.Users_df['Age'] = self.Users_df['bin']
    self.Users_df = self.Users_df.drop(columns='bin')
    self.Users_df = self.__convert2Categorical(self.Users_df, 'Age')
    self.Users_df = self.Users_df.drop(columns='Zip')
    return None
  
  def _prepare_data(self) -> None:
    users_df = pd.read_csv(
      os.path.join(self.movieLensDir, 'users.dat'),
      sep='\::',
      engine='python',
      names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
      dtype={
        'UID': 'uint16',
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
        'UID': 'uint16',
        'MID': 'uint16',
        'Rating': 'uint8',
        'Timestamp': 'uint64'
      }
    )

    # IMDB-ID to TMDB-ID is take too long so I will use TMDB-ID temporarily
    # ratings_df['MID'] = self.tmdb_api.get_tmdb_ids(ratings_df['MID'].values)
    
    self._prepare_pred_data(ratings_df=ratings_df, users_df=users_df)

    self.Users_df = users_df
    self.Ratings_df = ratings_df

    for occupation in OCCUPATION_MAP.items():
      self.Users_df['Occupation'] = self.Users_df['Occupation'].replace(occupation[1], occupation[0])

    self.__preprocess()

    self.GraphFeature = GraphFeature(self.Ratings_df, self.Users_df)
    self.GraphFeature_df = self.GraphFeature()

    self.whole_data = torch.Tensor(self.GraphFeature_df.values)
    
    self.whole_dataset = TensorDataset(self.whole_data)

    self.train_set, self.valid_set = random_split(self.whole_dataset, [(1 - self.val_rate), self.val_rate])
  
  def _prepare_pred_data(self, users_df: pd.DataFrame=None, ratings_df: pd.DataFrame=None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Movie Lens Dataset을 사용하는 이유가 Cold start problem 해결하기 위함이므로
    데이터가 충분히 많은 경우에는 Movie Lens Dataset을 사용하지 않고
    DataBaseLoader만을 사용해 데이터를 불러올 예정.
    '''
    assert self.dataBaseLoader is not None, 'GHRSDataset.dataBaseLoader must be set'
    assert users_df is not None, 'Users_df must be set'
    assert ratings_df is not None, 'Ratings_df must be set'
    
    db_ratings = self.dataBaseLoader.getReviews()
    ratings_df = pd.concat([ratings_df, db_ratings])
    db_users = self.dataBaseLoader.getUsers()
    ratings_df = pd.concat([ratings_df, db_users])
    return ratings_df, users_df

  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
  def val_dataloader(self):
    return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
  
  def predict_dataloader(self):
    return DataLoader(self.whole_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
  