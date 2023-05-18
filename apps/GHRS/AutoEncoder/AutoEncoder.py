import torch

import pytorch_lightning as pl

from torch import nn, Tensor
from copy import deepcopy
from typing import Tuple

class AutoEncoder(pl.LightningModule):
    def __init__(self, in_feature: int, latent_dim: int, learning_rate: float=0.05) -> None:
      super(AutoEncoder, self).__init__()
      self.in_feature = in_feature
      self.latent_dim = latent_dim
      self.learning_rate = learning_rate
      self.loss_fn = nn.MSELoss()

      assert (in_feature / 2) > latent_dim, 'Latent dimension must be less than half of input feature dimension'

      self._h_in_feature = int(in_feature / 2)

      self.Encoder = nn.Sequential(
        nn.Linear(in_feature, self._h_in_feature),
        nn.ReLU(),
        nn.Linear(self._h_in_feature, latent_dim),
      )

      self.Decoder = nn.Sequential(
        nn.Linear(latent_dim, self._h_in_feature),
        nn.ReLU(),
        nn.Linear(self._h_in_feature, in_feature),
        nn.Sigmoid(),
      )

    def _log_dict(self, log_dict):
      self.log_dict(
        dictionary=log_dict,
        prog_bar=True,
        on_step=False,
        on_epoch=True  
      )

    def _freezeLayer(self, layer: nn.Module) -> None:
      for param in layer.parameters():
        param.requires_grad = False

    def getFreezedEncoder(self) -> nn.Module:
       self._freezeLayer(self.Encoder)
       return deepcopy(self.Encoder)
    
    @torch.no_grad()
    def getLatnetVector(self, x: torch.Tensor) -> torch.Tensor:
      '''
      TODO Input data 에서의 UID와 Latent Vector를 맵핑하여 반환하도록 구현
      -> Tuple[UID, Latent Vector]
      '''
      self.Encoder.eval()
      latent_vec = self.Encoder(x)
      return dict(input=x, latent_vec=latent_vec)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
      encoded = self.Encoder(x)
      decoded = self.Decoder(encoded)
      return (encoded, decoded)

    def __common_step(self, batch: Tuple[Tensor, Tensor], batch_idx) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
      x = batch[0]
      x = x.reshape(x.size(0), -1)
      encoded, decoded = self.forward(x)
      loss = self.loss_fn(decoded, x)
      return loss, encoded, decoded
    
    def training_step(self, batch, batch_idx):
      loss, encoded, decoded = self.__common_step(batch, batch_idx)
      self._log_dict({'train_loss': loss})
      return loss
  
    def validation_step(self, batch, batch_idx):
      loss, encoded, decoded = self.__common_step(batch, batch_idx)
      self._log_dict({'valid_loss': loss})
      return loss
    
    def test_step(self, batch, batch_idx):
      loss, encoded, decoded = self.__common_step(batch, batch_idx)
      self._log_dict({'test_loss': loss})
      return loss

    def configure_optimizers(self):
      return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    