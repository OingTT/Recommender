import uvicorn

from apps.GHRS.GHRS import GHRS
from apps.arg_parser import get_args

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

cfgs = get_args()

ghrs = GHRS(CFG=cfgs)

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
# app.add_middleware(
#   HTTPSRedirectMiddleware
# )

@app.get('/')
async def root():
  return {}

@app.get(
  path='/recommendation/ottcomb/{uid}',
  description='Recommend OTT Combination',
)
async def recommend_ott_combination(uid: str):
  return ghrs.predict_ott_combination(UID=uid)

@app.get(
  path='/recommendation/ott/{uid}',
  description='Recommend OTT by subscription count'
)
async def recommend_ott(uid: str):
  return ghrs.predict_ott(UID=uid)

@app.get(
  path='/recommendation/{contentType}/{uid}',
  description='Recommend content => contentType = <MOVIE, TV>',
)
async def recommend_content(contentType: str, uid: str):
  return ghrs.predict_content(contentType=contentType, UID=uid)

if __name__=='__main__':
  uvicorn.run('app:app', host='0.0.0.0', port=10200, reload=False)