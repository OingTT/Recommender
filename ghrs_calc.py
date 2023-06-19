from time import sleep

from apps.GHRS.GHRSCalc import GHRSCalc
from apps.arg_parser import get_args


cfgs = get_args()

ghrs_calc = GHRSCalc(CFG=cfgs)

while True:
  sleep(60)
  try:
    ghrs_calc()
  except Exception as e:
    print(e)
    continue