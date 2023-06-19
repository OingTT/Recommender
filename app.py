import uvicorn

from threading import Thread

from apps.GHRS.GHRSPred import GHRSPred
from apps.GHRS.GHRSCalc import GHRSCalc
from apps.arg_parser import get_args

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

cfgs = get_args()

ghrs_pred = GHRSPred(CFG=cfgs)

app = FastAPI()

origins = [
  "*",
]

app.add_middleware(
  CORSMiddleware,
  allow_origins=origins,
  allow_credentials=True,
  allow_methods=["*"],
  allow_headers=["*"],
)

@app.get('/')
async def root():
  return {}

@app.get(
  path='/recommendation/ottcomb/{uid}',
  description='Recommend OTT Combination',
)
async def recommend_ott_combination(uid: str):
  return ghrs_pred.predict_ott_comb(target_UID=uid)

@app.get(
  path='/recommendation/ott/{uid}',
  description='Recommend OTT by subscription count'
)
async def recommend_ott(uid: str):
  return ghrs_pred.predict_ott(target_UID=uid)

@app.get(
  path='/recommendation/{contentType}/{uid}',
  description='Recommend content => contentType = <MOVIE, TV>',
)
async def recommend_content(contentType: str, uid: str):
  return ghrs_pred.predict_content(target_UID=uid, contentType=contentType)

if __name__=='__main__':
  uvicorn.run('app:app', host='0.0.0.0', port=10200, reload=False)