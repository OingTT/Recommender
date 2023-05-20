import wandb

from apps.GHRS.GHRS import GHRS
from apps.arg_parser import get_args


cfgs = get_args()

if not cfgs['debug']:
  wandb.login()

ghrs = GHRS(CFG=cfgs)
ghrs.trainAutoEncoder()

print("Predict Start")

from datetime import datetime

start = datetime.now()
prediction = ghrs.predict("33")
end = datetime.now()
print(end - start)
print(prediction)