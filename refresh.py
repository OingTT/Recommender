from time import sleep
from apps.GHRS.GHRS import GHRS
from apps.arg_parser import get_args


def refresh():
  print('refresh Called')
  
  cfgs = get_args()

  cfgs['refresh'] = True
  cfgs['train_ae'] = False
  ghrs = GHRS(CFG=cfgs)
  ghrs()

while True:
  try:
    refresh()
    sleep(10)
  except Exception:
    sleep(20)
    pass
