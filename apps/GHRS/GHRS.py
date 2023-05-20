import os
import wandb
import datetime

import pytorch_lightning as pl

from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger, WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from .Cluster.Cluster import Cluster
from .GHRSDataset import GHRSDataset
from .AutoEncoder.AutoEncoder import AutoEncoder

from apps.GHRS.DataBaseLoader import DataBaseLoader

class GHRS:
  def __init__(
      self,
      datasetDir: str = './ml-1m',
      CFG: dict = None
    ):
    '''
    TODO TMDB ID로 변경해서 보내야함
    https://github.com/OingTT/Recommender.git
    '''
    self.datasetDir = datasetDir
    self.debug = CFG['debug']
    self.train_AE = CFG['train_ae']
    self.latent_dim = CFG['latent_dim']
    self.batch_size = CFG['batch_size']
    self.num_workers = CFG['num_workers']
    self.val_rate = CFG['val_rate']
    self.accelerator = CFG['device']
    self.max_epoch = CFG['max_epoch']
    modelCheckpoint = ModelCheckpoint(
      save_top_k=10,
      monitor="valid_loss",
      mode="min",
      dirpath="./PretrainedModel",
      filename="Pretrained-{epoch:02d}-{valid_loss:.3f}",
    )
    self.ghrsDataset = GHRSDataset(
      self.datasetDir,
      DataBaseLoader=DataBaseLoader(),
      batch_size=self.batch_size,
      num_workers=self.num_workers,
      val_rate=self.val_rate
    )
    self.trainer = pl.Trainer(
      accelerator=self.accelerator,
      max_epochs=self.max_epoch,
      logger=self.__getLoggers(),
      callbacks=[modelCheckpoint],
      default_root_dir="./PretrainedModel",
      log_every_n_steps=1,
    )

  def __call__(self, ):
    '''
    this function assume that autoencoder is already trained
    '''
    chkps = dict()
    print(os.listdir("./PretrainedModel"))
    for chkp in os.listdir("./PretrainedModel"):
      if not chkp.startswith('Pretrained-epoch'):
        continue
      splited = chkp.split('-')
      epoch = int((splited[1]).split('=')[1])
      loss = float((splited[2]).split('=')[1].replace('.ckpt', ''))
      chkps.update({epoch: loss})
    min_loss = min(chkps.values())
    min_epoch = [epoch for epoch, loss in chkps.items() if loss == min_loss][0]
    best_model = f'PretrainedModel/Pretrained-epoch={min_epoch}-valid_loss={min_loss}.ckpt'

    self.autoEncoder = AutoEncoder.load_from_checkpoint(
      checkpoint_path=best_model,
    )
    prediction = self.trainer.predict(self.autoEncoder, datamodule=self.ghrsDataset)
    return prediction
    
  def __getLoggers(self) -> list:
    log_dir = f'./train_logs'
    modelName = f'GHRS_{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
    loggers = list()
    loggers.append(CSVLogger(
      save_dir=log_dir,
      name=modelName,
    ))
    if not self.debug:
      loggers.append(TensorBoardLogger(
        save_dir=log_dir,
        name=modelName,
        log_graph=True,
      ))
      loggers.append(WandbLogger(
        project=f'GHRS',
        name=modelName,
      ))
    return loggers

  def trainAutoEncoder(self):
    self.autoEncoder = AutoEncoder(len(self.ghrsDataset), self.latent_dim)
    self.trainer.fit(
      self.autoEncoder,
      datamodule=self.ghrsDataset,
    )
  
  def exportModel(self, model: pl.LightningModule, modelName: str = 'GHRS') -> None:
    '''
    Export model to outside of docker container
    '''
    model.save_checkpoint(f'./PretrainedModel/{modelName}.ckpt')

  def predictAutoencoder(self):
    return self.trainer.predict(self.autoEncoder, self.ghrsDataset)
  
  def cluster(self):
    self.cluster = Cluster(self.graphFeatureDF)
    return self.cluster()
