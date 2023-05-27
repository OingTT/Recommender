import apps.GHRS.GHRS as GHRS
from fastapi import APIRouter, Depends

rounter = APIRouter(
    prefix='recommendation/movie',
    dependencies=Depends(GHRS)
)