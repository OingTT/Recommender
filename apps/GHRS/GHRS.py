import datetime
import wandb

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger

from .Cluster.Cluster import Cluster
from .GHRS_Dataset import GHRS_Dataset
from .AutoEncoder.AutoEncoder import AutoEncoder


class GHRS:
  def __init__(
      self,
      datasetDir: str = './ml-1m',
      train_AE: bool=True,
      latent_dim: int=8,
      batch_size: int=1024,
      num_workers: int=8,
      val_rate: float=0.2,
    ):
    self.datasetDir = datasetDir
    self.train_AE = train_AE
    self.latent_dim = latent_dim
    self.batch_size = batch_size
    self.num_workers = num_workers
    self.val_rate = val_rate
    self.trainer = pl.Trainer(
      accelerator='gpu',
      max_epochs=100,
      logger=self.__getLoggers(),
    )

  def __call__(self, ):
    '''
    this function assume that autoencoder is already trained
    '''
    self.ghrsDataset
    self.trainer.predict(self.autoEncoder, self.ghrsDataset)
    
  def __getLoggers(self, ):
    log_dir = f'./train_logs'
    modelName = f'GHRS_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    tb_logger = TensorBoardLogger(
      save_dir=log_dir,
      name=modelName,
      log_graph=True,
    )
    csv_logger = CSVLogger(
      save_dir=log_dir,
      name=modelName,
    )
    wandb_logger = WandbLogger(
      project=f'GHRS',
      name=modelName,
    )
    return [tb_logger, csv_logger, wandb_logger]

  def trainAutoEncoder(self):
    self.ghrsDataset = GHRS_Dataset(self.datasetDir, batch_size=self.batch_size, num_workers=self.num_workers, val_rate=self.val_rate)
    self.autoEncoder = AutoEncoder(len(self.ghrsDataset), self.latent_dim)
    self.trainer.fit(
      self.autoEncoder,
      datamodule=self.ghrsDataset,
    )
    
  def predictAutoencoder(self, ):
    return self.trainer.predict(self.autoEncoder, self.ghrsDataset)
  
  def cluster(self, ):
    self.cluster = Cluster(self.graphFeatureDF)
    return self.cluster()
