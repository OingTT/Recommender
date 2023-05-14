import wandb

from apps.GHRS.GHRS import GHRS

wandb.login()

ghrs = GHRS()
ghrs.trainAutoEncoder()