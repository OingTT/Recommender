import os

import pandas as pd

from tqdm import tqdm
from apps.apis.TMDB import TMDB
from apps.GHRS.MovieLens import MovieLens

if __name__=='__main__':
  ratings_df = pd.read_csv(
    os.path.join('./ml-1m', 'ratings.dat'),
    sep='\::',
    engine='python',
    names=['UID', 'MID', 'Rating', 'Timestamp'],
    dtype={
      'UID': 'uint8',
      'MID': 'uint16',
      'Rating': 'uint8',
      'Timestamp': 'uint64'
    }
  )
  tmdb_api = TMDB()

  imdb2Tmdb = dict()
  ratings_grouped = ratings_df['MID'].groupby(ratings_df['MID'])
  for mid, group in tqdm(ratings_grouped.groups.items()):
    if mid in imdb2Tmdb.keys():
      continue
    tmdb_id = tmdb_api.get_tmdb_id(mid)
    print(mid, tmdb_id)
    imdb2Tmdb['IMDB-ID'] = mid
    imdb2Tmdb['TMDB-ID'] = tmdb_id
  print(imdb2Tmdb)

  df = pd.DataFrame.from_dict(imdb2Tmdb, columns=['IMDB-ID', 'TMDB-ID'], orient='index')

  df.to_csv('./imdb2tmdb.csv', index=False)