import wandb

from apps.GHRS.GHRS import GHRS
from apps.arg_parser import get_args


# cfgs = get_args()

# # if not cfgs['debug']:
# #   wandb.login()

# # ghrs = GHRS(CFG=cfgs)
# # ghrs.trainAutoEncoder()

# # print("Predict Start")

# # from datetime import datetime

# # # start = datetime.now()
# # # ghrs = GHRS(CFG=cfgs)
# # # prediction = ghrs.predict("clhkairuv0000mn08gt2yvi5b")
# # # end = datetime.now()
# # # print(end - start)
# # # print(prediction)

# # ghrs = GHRS(CFG=cfgs)

# # res = ghrs.predict_ott_combination('clhkairuv0000mn08gt2yvi5b')

# # print(res)

# from apps.GHRS.GHRSCalc import GHRSCalc

# cg = GHRSCalc(cfgs)
# cg()

import apps.database.models