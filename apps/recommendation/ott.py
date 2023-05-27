import apps.GHRS.GHRS as GHRS
from fastapi import APIRouter, Depends

rounter = APIRouter(
    prefix='recommendation/ott',
    dependencies=Depends(GHRS)
)