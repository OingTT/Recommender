
from apps.arg_parser import get_args
from apps.GHRS.GHRS import GHRS

cfg = get_args()

ghrs = GHRS(CFG=cfg)
ghrs.trainAutoEncoder()