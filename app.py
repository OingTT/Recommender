from apps.arg_parser import get_args
from apps.GHRS.GHRS import GHRS
from fastapi import FastAPI

app = FastAPI()

cfgs = {
    'debug': False,
    'log': False,
    'device': 'gpu',
    'train_ae': False,
    'latent_dim': 8,
    'batch_size': 1024,
    'num_workers': 4,
    'val_rate': 0.2,
    'max_epoch': 100,
    'sample_rate': 0,
}

@app.get("/recommendation/{uid}")
async def recommend_movie(uid):
    ghrs = GHRS(CFG=cfgs)

    return ghrs.predict(uid)