import os
import torch

import numpy as np
import pandas as pd
import pytorch_lightning as pl

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import TensorDataset, DataLoader, random_split

from .Const import OCCUPATION_MAP
from .DataBaseLoader import DataBaseLoader
from .GraphFeature.GraphFeature_GraphTool import GraphFeature_GraphTool

class GHRS_Dataset(pl.LightningDataModule):
  def __init__(self, movieLensDir: str = './ml-1m', DataBaseLoader: DataBaseLoader=None, batch_size: int = 64, num_workers: int = 0, val_rate: float=0.2) -> None:
    super(GHRS_Dataset, self).__init__()
    self.movieLensDir = movieLensDir
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.val_rate = val_rate
    self.dataBaseLoader = DataBaseLoader
    self._prepare_data()

  def __len__(self):
    '''
    Return dim of datasets feature
    '''
    return len(self.GraphFeature_df.loc[:, ~self.GraphFeature_df.columns.isin(['Rating', 'UID', 'MID'])].columns)

  def __convert2Categorical(self, df_X: pd.DataFrame, _X: str):
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

  def __preprocess(self):
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
    return
  
  def _prepare_data(self) -> None:
    self.Ratings_df = pd.read_csv(
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
    self.Users_df = pd.read_csv(
      os.path.join(self.movieLensDir, 'users.dat'),
      sep='\::',
      engine='python',
      names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
      dtype={
        'UId': 'uint16',
        'Gender': 'str',
        'Age': 'uint8',
        'Occupation': 'uint8',
        'Zip': 'string'
      }
    )
    for occupation in OCCUPATION_MAP.items():
      self.Users_df['Occupation'] = self.Users_df['Occupation'].replace(occupation[1], occupation[0])

    self.__preprocess()

    self.GraphFeature = GraphFeature_GraphTool(self.Ratings_df, self.Users_df)
    self.GraphFeature_df: pd.DataFrame = self.GraphFeature()

    self.whole_data = torch.Tensor(
      self.GraphFeature_df.loc[:, ~self.GraphFeature_df.columns.isin(['Rating', 'UID', 'MID'])].values
    )
    
    self.whole_dataset = TensorDataset(self.whole_data)

    self.train_set, self.valid_set = random_split(self.whole_dataset, [(1 - self.val_rate), self.val_rate])

  def train_dataloader(self):
    return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
  
  def val_dataloader(self):
    return DataLoader(self.valid_set, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False)
  