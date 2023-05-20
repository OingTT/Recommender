import wandb

from apps.GHRS.GHRS import GHRS
from arg_parser import get_args


cfgs = get_args()

if not cfgs['debug']:
  wandb.login()

ghrs = GHRS(CFG=cfgs)
ghrs.trainAutoEncoder()

prediction = ghrs()

print(prediction)