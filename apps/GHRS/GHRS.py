import os
import wandb
import datetime

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .Cluster.Cluster import Cluster
from .GHRSDataset import GHRS_Dataset
from .AutoEncoder.AutoEncoder import AutoEncoder


class GHRS:
  def __init__(
      self,
      datasetDir: str = './ml-1m',
      CFG: dict = None
      # train_AE: bool=True,
      # latent_dim: int=8,
      # batch_size: int=1024,
      # num_workers: int=8,
      # val_rate: float=0.2,
    ):
    self.datasetDir = datasetDir
    self.train_AE = CFG['train_ae']
    self.latent_dim = CFG['latent_dim']
    self.batch_size = CFG['batch_size']
    self.num_workers = CFG['num_workers']
    self.val_rate = CFG['val_rate']
    self.accelerator = CFG['device']
    self.max_epoch = CFG['max_epoch']
    modelCheckpoint = ModelCheckpoint(
      save_top_k=10,
      monitor="val_loss",
      mode="min",
      dirpath="./PretrainedModel",
      filename="Pretrained-{epoch:02d}-{val_loss:.2f}",
    )
    self.trainer = pl.Trainer(
      accelerator=self.accelerator,
      max_epochs=self.max_epoch,
      logger=self.__getLoggers(),
      callbacks=[modelCheckpoint],
      default_root_dir="./PretrainedModel",
    )

  def __call__(self, ):
    '''
    this function assume that autoencoder is already trained
    '''
    chkps = dict()
    for chkp in os.listdir():
      if not chkp.startswith('Pretrained-epoch'):
        continue
      splited = chkp.split('-')
      epoch = int(splited[1])
      loss = float(splited[2])
      chkps.update({epoch: loss})
    min_loss = min(chkps.values())
    min_epoch = [epoch for epoch, loss in chkps.items() if loss == min_loss][0]
    best_model = f'Pretrained-{min_epoch}-{min_loss:.2f}.ckpt'

    self.autoEncoder = AutoEncoder.load_from_checkpoint(
      checkpoint_path=best_model,
    )
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
