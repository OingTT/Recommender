
from apps.arg_parser import get_args
from apps.GHRS.GHRS import GHRS

cfg = get_args()

cfg['debug'] = True
cfg['log'] = True
cfg['device'] = 'gpu'
cfg['max_epoch'] = 100
cfg['sample_rate'] = 1

ghrs = GHRS(CFG=cfg)
ghrs.train()