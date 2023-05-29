import uvicorn

from apps.GHRS.GHRS import GHRS
from apps.arg_parser import get_args

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

@app.get("/recommendation/{contentType}/{uid}")
async def recommend_movie(contentType: str, uid: str):
    return ghrs.predict_content(contentType=contentType, UID=uid)

@app.get("/recommendation/ott/{uid}")
async def recommend_ott(uid: str):
    return ghrs.predict_ott(uid)

if __name__=='__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8888, reload=True)