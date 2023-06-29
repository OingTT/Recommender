from traceback import print_tb

from time import sleep
from apps.GHRS.GHRSCalc import GHRSCalc
from apps.arg_parser import get_args


cfgs = get_args()

cfgs['debug'] = True
cfgs['optimal_k_method'] = 'Silhouette'

ghrs_calc = GHRSCalc(CFG=cfgs)

while True:
  try:
    ghrs_calc()
  except Exception as e:
    print(e)
    print_tb(e.__traceback__)
    continue
  sleep(60)