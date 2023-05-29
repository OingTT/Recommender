import os
import wandb
import datetime

import pandas as pd
import pytorch_lightning as pl

from typing import Tuple, List
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from apps.Singleton import Singleton
from apps.GHRS.Cluster.Cluster import Cluster
from apps.GHRS.Dataset.GHRSDataset import GHRSDataset
from apps.GHRS.AutoEncoder.AutoEncoder import AutoEncoder
from apps.GHRS.Dataset.DataBaseLoader import DataBaseLoader

class GHRS(metaclass=Singleton):
  '''
  TODO Training Step과 Prediction Step을 분리해야함
  __call__ => Prediction & Recommendation
  training => Train Autoencoder & Save Best model to pretrained_models
  '''
  autoencoder = None

  def __init__(
      self,
      CFG: dict = None
    ):
    self.CFG = CFG
    self.cluster = Cluster()
    self.databaseLoader = DataBaseLoader()
    self.datasetDir = self.CFG['movie_lens_dir']
    modelCheckpoint = self.__init_model_checkpoint()
    self.ghrsDataset = GHRSDataset(
      CFG=self.CFG,
      movieLensDir=self.datasetDir,
      DataBaseLoader=self.databaseLoader,
    )
    loggers = self.__init_loggers()
    self.trainer = pl.Trainer(
      accelerator=self.CFG['device'],
      max_epochs=self.CFG['max_epoch'],
      logger=loggers,
      callbacks=modelCheckpoint,
      default_root_dir="./pretrained_model",
      log_every_n_steps=1,
    )

  def __delete_model_check_points(self) -> None:
    pretrained_model_path = self.CFG['pretrained_model_dir']
    if len(os.listdir(pretrained_model_path)) != 0:
      for item in os.listdir(pretrained_model_path):
        if os.path.isfile(os.path.join(pretrained_model_path, item)):
          os.remove(os.path.join(pretrained_model_path, item))
  
  def __init_model_checkpoint(self) -> List[ModelCheckpoint]:
    pretrained_model_path = self.CFG['pretrained_model_dir']
    if self.CFG['train_ae']:
      self.__delete_model_check_points()
    return [ModelCheckpoint(
      save_top_k=10,
      monitor="valid_loss",
      mode="min",
      dirpath=pretrained_model_path,
      filename="Pretrained-{epoch:02d}-{valid_loss:.4f}",
    )]
  
  def __load_best_model(self) -> pl.LightningModule:
    chkps = dict()
    pretrained_model_dir: str = self.CFG['pretrained_model_dir']
    if not os.path.exists(pretrained_model_dir):
      os.mkdir(pretrained_model_dir)
    for chkp in os.listdir(pretrained_model_dir):
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

    best_model = os.path.join(pretrained_model_dir, 'Pretrained-epoch={min_epoch}-valid_loss={min_loss:.4f}.ckpt')

    return AutoEncoder.load_from_checkpoint(
      checkpoint_path=best_model,
    )

  def __init_loggers(self) -> List:
    if not self.CFG['log']:
      return list()
    log_dir = self.CFG['log_dir']
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

  def __call__(self, UID: str) -> Tuple[pd.DataFrame, ...]:
    '''
    this function assume that autoencoder is already trained
    '''
    ...

  def __get_latent_df(self, model: pl.LightningModule, graphFeature_df: pd.DataFrame) -> pd.DataFrame:
    if model is None:
      raise ValueError('model is None')
    # prediction => prediction of autoencoder for each batch
    latent_df: list[pd.DataFrame] = self.trainer.predict(model, datamodule=self.ghrsDataset)
    # prediction => prediction of autoencoder for all data
    latent_df: pd.DataFrame = pd.concat(latent_df, axis=0)
    return latent_df
  
  def __clustering(self, encoded_df: pd.DataFrame) -> pd.DataFrame:
    '''
    return DataFrame which include cluster labels
    '''
    return self.cluster(encoded_df=encoded_df)
  
  def __drop_movie_lens(self, df: pd.DataFrame) -> pd.DataFrame:
    '''
    While training, we use movie lens data
    but while prediction, we don't use movie lens data
    '''
    return df[~df['UID'].isin(self.ghrsDataset.users_df['UID'])]
  
  def __get_target_cluster_uids(self, UID: str) -> List[str]:
    if self.autoencoder is None:
      self.autoencoder = self.__load_best_model()

    # prediction => prediction of autoencoder
    prediction = self.__get_latent_df(self.autoencoder, self.ghrsDataset.GraphFeature_df)

    # clustered => prediction with cluster label
    clustered = self.__clustering(encoded_df=prediction)

    # clustered => concat with UID
    clustered = pd.concat([self.ghrsDataset.GraphFeature_df['UID'], clustered], axis=1)

    # clustered => drop movie lens data
    clustered = self.__drop_movie_lens(clustered)

    # grouped => grouped by cluster label
    grouped = clustered.groupby('cluster_label', as_index=False)

    # target_cluster => cluster label that UID is in
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

    # target_cluster_uids => uids of target cluster
    target_cluster_uids: List[str] = map(str, target_cluster_df['UID'].values.tolist())

    return target_cluster_uids

  def predict_content(self, contentType: str, UID: str, topN: int=20) -> List[dict]:
    '''
    return Dataframe of mean rating about target user's cluster
    columns: ['ContentType', 'CID', 'Rating']
    '''
    target_cluster_uids = self.__get_target_cluster_uids(UID=UID)

    # target_cluster_rating => ratings about every contentType of target cluster
    target_cluster_rating = self.ghrsDataset.ratings_df[self.ghrsDataset.ratings_df['UID'].isin(target_cluster_uids)]

    # target_cluster_rating => ratings about specified contentType of target cluster
    target_cluster_rating = target_cluster_rating[target_cluster_rating['ContentType'] == contentType]

    # target_cluster_rating_mean => mean rating of each content in target cluster
    target_cluster_rating_mean = target_cluster_rating.groupby('CID', as_index=False)['Rating'].mean()

    # target_user_rating => ratings of target user
    target_user_rating = self.ghrsDataset.ratings_df[self.ghrsDataset.ratings_df['UID'] == UID]['CID'].dropna().values.tolist()

    # target_cluster_rating_mean => drop movies that target user already watched
    target_cluster_rating_mean = \
      target_cluster_rating_mean[~target_cluster_rating_mean['CID'].isin(target_user_rating)]
    
    # target_cluster_rating_mean_rating => sort by mean rating
    target_cluster_rating_mean.sort_values(by='Rating', ascending=False, inplace=True)
    
    # target_cluster_rating_mean_rating => top 10 movies
    target_cluster_rating_mean = target_cluster_rating_mean.iloc[: topN]

    return self.__content_prediction_to_json(target_cluster_rating_mean)
  
  def predict_ott(self, UID: str, topN: int=20) -> List[dict]:
    # target_cluster_uids => uids of target cluster
    target_cluster_uids = self.__get_target_cluster_uids(UID=UID)

    # subscription => subscription of target cluster
    subscription = self.databaseLoader.getAllSubscription()

    # subscription => subscription of target cluster
    subscription = subscription[subscription['UID'].isin(target_cluster_uids)]
    
    # subscription => group by OTT & count
    subscription = subscription.groupby('OTT', as_index=False)['UID'].count()

    # subscription => sort by subscription count
    subscription.sort_values(by='UID', ascending=False, inplace=True)

    subscription = subscription.iloc[: topN]

    return self.__ott_prediction_to_json(subscription)

  def __content_prediction_to_json(self, rating_mean: pd.DataFrame) -> List[dict]:
    results = list()
    for i in range(len(rating_mean)):
      contentType = rating_mean['ContentType'].iloc[i]
      cid = rating_mean['CID'].iloc[i]
      rating = rating_mean['Rating'].iloc[i]
      results.append(dict(
        ContentType=contentType,
        ContentID=int(cid),
        Rating=int(rating)
      ))
    return results
  
  def __ott_prediction_to_json(self, subscription_count: pd.DataFrame) -> List[dict]:
    results = list()
    for i in range(len(subscription_count)):
      ott = subscription_count['OTT'].iloc[i]
      count = subscription_count['UID'].iloc[i]
      results.append(dict(
        OTT=ott,
        Count=int(count)
      ))
    return results
  
  def train(self):
    '''
    Train AutoEncoder model: Use MovieLens, DB dataset
    '''
    self.autoencoder = AutoEncoder(len(self.ghrsDataset), self.CFG['latent_dim'])
    self.trainer.fit(
      self.autoencoder,
      datamodule=self.ghrsDataset,
    )
  