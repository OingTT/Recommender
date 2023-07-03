import os

import pandas as pd
import pytorch_lightning as pl

from datetime import datetime
from typing import Tuple, List
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger, Logger

from apps.database.DatabaseAdapter import DatabaseAdapter
from apps.utils.Singleton import Singleton
from apps.GHRS.Dataset.DataBaseLoader import DataBaseLoader
from apps.GHRS.Dataset.MovieLensLoader import MovieLensLoader
from apps.GHRS.Dataset.GHRSDataset import GHRSDataset
from apps.GHRS.AutoEncoder.AutoEncoder import AutoEncoder
from apps.GHRS.Clustering.Clustering import Clustering

from apps.GHRS.GraphFeature.GraphFeature_GraphTool \
  import GraphFeature_GraphTool \
    as GraphFeature

class GHRSCalc(metaclass=Singleton):
  '''
  Clueter-Matrix 계산 및 저장
  1. Database에 있는 유저, 평점 정보 불러오기 
      (Autuencoder 훈련 시에는 MovieLens포함) -> DataFrame
  2. Graph Feature 구하기 -> DataFrame
  3. TorchDataset으로 변환 -> TensorDataset
  4. Auto Encoder 입력 -> DataFrame
  5. Clustering -> DataFrame
  '''
  def __init__(self, CFG: dict):
    self.CFG = CFG
    self.databaseAdapter = DatabaseAdapter(CFG=self.CFG)
    self.databaseLoader = DataBaseLoader(databaseAdapter=self.databaseAdapter)
    self.movielensLoader = MovieLensLoader(CFG=self.CFG)
    self.clustering = Clustering(self.CFG)

  def __call__(self):
    self.graphFeature = GraphFeature(CFG=self.CFG)

    users, ratings = self.__get_data()
    
    graph_features = self.__get_graph_features(users, ratings)

    ghrs_datamodule = GHRSDataset(self.CFG, graph_features)
    
    autoencoder = self.__load_best_model()

    autoencoder_trainer = pl.Trainer(
      accelerator=self.CFG['device'],
      logger=self.__init_loggers(),
      default_root_dir="./pretrained_model",
      log_every_n_steps=1,
    )

    latent_matrix = self.__get_latent_matrix(autoencoder, autoencoder_trainer, ghrs_datamodule)

    clustered = self.__get_clusters(self.clustering, latent_matrix)
    
    clustered = pd.concat([users['id'], pd.Series(clustered, name='cluster_label')], axis=1)

    self.__store_clustered(clustered)

  def __get_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    db_data = self.__get_db_data()
    if not db_data:
      raise ValueError('Cannot get data from database')
    else:
      users, ratings = db_data

    if self.CFG['ml_sample_rate'] != 0:
      ml_data = self.__get_ml_data()
      if not ml_data:
        raise ValueError('Cannot get data from movielens')
      else:
        ml_users, ml_ratings = ml_data
        users = pd.concat([users, ml_users], axis=0)
        ratings = pd.concat([ratings, ml_ratings], axis=0)
    
    return users, ratings

  def __store_clustered(self, clustered: pd.DataFrame):
    for _, row in clustered.iterrows():
      id = str(row['id'])
      label = int(row['cluster_label'])
      self.databaseAdapter.updateUserClusteredByUserId(id=id, label=label)

  def __load_best_model(self) -> AutoEncoder:
    chkps = dict()
    pretrained_model_dir: str = self.CFG['pretrained_model_dir']

    model_dir_list = os.listdir(pretrained_model_dir)

    if len(model_dir_list) == 0:
      raise FileNotFoundError('There is no pretrained model in pretrained_model_dir')

    for chkp in model_dir_list:
      if not chkp.startswith('Pretrained-epoch'):
        continue
      splited = chkp.split('-')
      epoch = int((splited[1]).split('=')[1])
      loss = float((splited[2]).split('=')[1].replace('.ckpt', ''))
      chkps.update({epoch: loss})
    min_loss = min(chkps.values())
    min_epoch = [epoch for epoch, loss in chkps.items() if loss == min_loss][0]
    if len(str(min_epoch)) == 1:
      min_epoch = f'0{min_epoch}'

    best_model = os.path.join(pretrained_model_dir, f'Pretrained-epoch={min_epoch}-valid_loss={min_loss:.4f}.ckpt')

    return AutoEncoder.load_from_checkpoint(
      checkpoint_path=best_model,
    )
  
  def __init_loggers(self) -> List[Logger]:
    if not self.CFG['log']:
      return False
    log_dir = self.CFG['log_dir']
    modelName = f'GHRS_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    return [
      CSVLogger(save_dir=log_dir, name=modelName),
      # TensorBoardLogger(save_dir=log_dir, name=modelName, log_graph=True),
      # WandbLogger(project=f'GHRS', name=modelName)
    ]
  
  def __get_db_data(self) -> Tuple[pd.DataFrame, pd.DataFrame] | bool:
    users = self.databaseLoader.getAllUsers()
    ratings = self.databaseLoader.getAllReviews()
    if users.empty or ratings.empty:
      return False
    return users, ratings

  def __get_ml_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users, ratings = self.movielensLoader()
    if users.empty or ratings.empty:
      return False
    return users, ratings

  def __get_graph_features(self, users: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    graph_features = self.graphFeature.calculate(users, ratings, self.CFG['alpha_coefficient'])
    return graph_features
  
  def __get_latent_matrix(
      self,
      autoencoder: pl.LightningModule,
      autoencoder_trainer: pl.Trainer,
      ghrs_datamodule: pl.LightningDataModule
    ) -> pd.DataFrame:
    if not isinstance(autoencoder, pl.LightningModule):
      raise ValueError('autoencoder is not instance of pl.LightningModule')
    
    # latent_matrix => prediction of autoencoder for each batch
    latent_matrix: list[pd.DataFrame] = autoencoder_trainer.predict(autoencoder, datamodule=ghrs_datamodule)
    # latent_matrix => prediction of autoencoder for all data
    latent_matrix: pd.DataFrame = pd.concat(latent_matrix, axis=0)

    return latent_matrix

  def __get_clusters(self, clustering: Clustering, latent_matrix: pd.DataFrame):
    return clustering(latent_matrix)