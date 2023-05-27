import uvicorn

from apps.GHRS.GHRS import GHRS

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



ghrs = GHRS(CFG=cfgs)
@app.get("/recommendation/{uid}")
async def recommend_movie(uid):
    return ghrs.predict(uid)

if __name__=='__main__':
    uvicorn.run('app:app', host='0.0.0.0', port=8888, reload=True)