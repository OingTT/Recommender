import os
import wandb
import datetime

import pandas as pd
import pytorch_lightning as pl

from typing import Tuple, List
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from apps.utils.utils import print_log
from apps.GHRS.GHRSDataset import GHRSDataset
from apps.GHRS.Cluster.Cluster import Cluster
from apps.GHRS.DataBaseLoader import DataBaseLoader
from apps.GHRS.AutoEncoder.AutoEncoder import AutoEncoder

class GHRS:
  '''
  TODO Training Step과 Prediction Step을 분리해야함
  TODO Prediction Step에서는 UID를 입력받아서 해당 유저의 대한 추천 컨텐츠를 반환해야함
  TODO torch.Tensor는 string값을 가질 수 없음
  '''
  def __init__(
      self,
      datasetDir: str = './ml-1m',
      CFG: dict = None
    ):
    '''
    TODO TMDB ID로 변경해서 보내야함
    '''
    self.datasetDir = datasetDir
    self.CFG = CFG
    modelCheckpoint = ModelCheckpoint(
      save_top_k=10,
      monitor="valid_loss",
      mode="min",
      dirpath="./pretrained_model",
      filename="Pretrained-{epoch:02d}-{valid_loss:.4f}",
    )
    self.ghrsDataset = GHRSDataset(
      CFG=self.CFG,
      movieLensDir=self.datasetDir,
      DataBaseLoader=DataBaseLoader(),
    )
    loggers = self.__getLoggers()
    self.trainer = pl.Trainer(
      accelerator=self.CFG['device'],
      max_epochs=self.CFG['max_epoch'],
      logger=loggers,
      callbacks=[modelCheckpoint],
      default_root_dir="./PretrainedModel",
      log_every_n_steps=1,
    )

  def __load_best_model(self):
    chkps = dict()
    for chkp in os.listdir("./pretrained_model"):
      if not chkp.startswith('Pretrained-epoch'):
        continue
      splited = chkp.split('-')
      epoch = int((splited[1]).split('=')[1])
      loss = float((splited[2]).split('=')[1].replace('.ckpt', ''))
      chkps.update({epoch: loss})
    min_loss = min(chkps.values())
    min_epoch = [epoch for epoch, loss in chkps.items() if loss == min_loss][0]
    print(min_epoch)
    if len(str(min_epoch)) == 1:
      min_epoch = f'0{min_epoch}'
    print(min_epoch)
    best_model = f'./pretrained_model/Pretrained-epoch={min_epoch}-valid_loss={min_loss:.4f}.ckpt'

    return AutoEncoder.load_from_checkpoint(
      checkpoint_path=best_model,
    )

  def __call__(self) -> Tuple[pd.DataFrame, ...]:
    '''
    this function assume that autoencoder is already trained
    '''
    autoEncoder = self.__load_best_model()
    prediction: list = self.trainer.predict(autoEncoder, datamodule=self.ghrsDataset)
    prediction: pd.DataFrame = pd.concat(prediction, axis=0)
    clustered = self.cluster(encoded_df=prediction)
    grouped = clustered.groupby('cluster_label', as_index=False)
    return grouped.groups
  
  def predict(self, UID: str) -> pd.DataFrame:
    '''
    return dataframe of recommended movies
    with columns ['MID', 'Rating']
    '''
    autoEncoder = self.__load_best_model()

    # prediction => prediction of autoencoder for each batch
    prediction: list = self.trainer.predict(autoEncoder, datamodule=self.ghrsDataset)
    # prediction => prediction of autoencoder for all data
    prediction: pd.DataFrame = pd.concat(prediction, axis=0)
    # clustered => prediction with cluster label
    clustered = self.cluster(encoded_df=prediction)
    # grouped => grouped by cluster label
    grouped = clustered.groupby('cluster_label', as_index=False)

    # target_cluster => cluster that UID is in
    target_cluster = None
    for cluster_label, indices in grouped.groups.items():
      user_in_cluster = self.ghrsDataset.users_df.iloc[indices]['UID'].values.tolist()
      if UID in user_in_cluster:
        target_cluster = cluster_label
        break
    if target_cluster is None:
      raise ValueError('UID is not in any cluster')

    # target_cluster_df => dataframe of target cluster
    target_cluster_df = clustered[clustered['cluster_label'] == target_cluster]
    print(target_cluster_df)

    # target_cluster_uids => uids of target cluster
    target_cluster_uids: List[str] = map(str, map(int, target_cluster_df['UID'].values.tolist()))
    print(target_cluster_uids)

    # target_cluster_rating => ratings of target cluster
    target_cluster_rating = self.ghrsDataset.ratings_df[self.ghrsDataset.ratings_df['UID'].isin(target_cluster_uids)]
    print(target_cluster_rating)

    # target_cluster_rating_mean => mean rating of each movie in target cluster
    target_cluster_rating_mean = target_cluster_rating.groupby('MID', as_index=False)['Rating'].mean()
    print(target_cluster_rating_mean)

    # target_user_rating => ratings of target user
    target_user_rating = self.ghrsDataset.ratings_df[self.ghrsDataset.ratings_df['UID'] == UID]['MID'].dropna().values.tolist()
    print(target_user_rating)

    # target_cluster_rating_mean => drop movies that target user already watched
    target_cluster_rating_mean = \
      target_cluster_rating_mean[~target_cluster_rating_mean['MID'].isin(target_user_rating)]
    print(target_cluster_rating_mean)
    
    # target_cluster_rating_mean_rating => sort by mean rating
    target_cluster_rating_mean.sort_values(by='Rating', ascending=False, inplace=True)
    
    # target_cluster_rating_mean_rating => top 10 movies
    target_cluster_rating_mean = target_cluster_rating_mean.iloc[:10]
    print_log(self.CFG, values=target_cluster_rating_mean)

    return target_cluster_rating_mean
    
  def __getLoggers(self) -> list:
    if not self.CFG['log']:
      return list()

    log_dir = f'./train_logs'
    modelName = f'GHRS_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    return [
      CSVLogger(
        save_dir=log_dir,
        name=modelName,
      ),
      TensorBoardLogger(
        save_dir=log_dir,
        name=modelName,
        log_graph=True,
      ),
      WandbLogger(
        project=f'GHRS',
        name=modelName,
      )
    ]

  def trainAutoEncoder(self):
    self.autoEncoder = AutoEncoder(len(self.ghrsDataset), self.CFG['latent_dim'])
    self.trainer.fit(
      self.autoEncoder,
      datamodule=self.ghrsDataset,
    )
  
  # def exportModel(self, model: pl.LightningModule, modelName: str = 'GHRS') -> None:
  #   '''
  #   Export model to outside of docker container
  #   '''
  #   model.save_checkpoint(f'./pretrained_model/{modelName}.ckpt')

  def predictAutoencoder(self):
    return self.trainer.predict(self.autoEncoder, self.ghrsDataset)
  
  def cluster(self, encoded_df: pd.DataFrame):
    cluster = Cluster()
    cluster_label = cluster(encoded_df=encoded_df)
    encoded_df['cluster_label'] = cluster_label
    return encoded_df
