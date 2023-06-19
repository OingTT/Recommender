import os
import pandas as pd

from typing import Tuple

class MovieLensLoader:
  def __init__(self, CFG: dict) -> None:
    self.CFG = CFG

  def __call__(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
    users = self.__get_ml_users()
    ratings = self.__get_ml_ratings()

    sample_rate = self.CFG.get('ml_smple_rate', 0)
    if sample_rate != 1:
      random_state = self.CFG.get('random_state', 1)
      self.__movie_lens_sampler(users, ratings, sample_rate, random_state)

    return users, ratings  

  def __movie_lens_sampler(self,
                          users_df: pd.DataFrame,
                          ratings_df: pd.DataFrame,
                          sample_rate: float,
                          random_state: int
                          ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    sample_len = int(len(users_df) * sample_rate)
    sampled_users_df = users_df.sample(sample_len, random_state=random_state)
    sampled_ratings_df = ratings_df[ratings_df['UID'].isin(sampled_users_df['UID'])]
    return sampled_users_df, sampled_ratings_df
  
  def __get_ml_users(self) -> pd.DataFrame:
    users_path = os.path.join(self.CFG['movielens_dir'], 'users.dat')
    users = pd.read_csv(
      users_path,
      sep='\::',
      engine='python',
      names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
      dtype={
        'UID': 'str',
        'Gender': 'str',
        'Age': 'uint8',
        'Occupation': 'uint8',
        'Zip': 'string'
      }
    )
    return users
  
  def __get_ml_ratings(self) -> pd.DataFrame:
    ratings_path = os.path.join(self.CFG['movielens_dir'], 'ratings.dat')
    ratings = pd.read_csv(
      filepath_or_buffer=ratings_path,
      sep='\::',
      engine='python',
      names=['UID', 'CID', 'Rating', 'Timestamp'],
      dtype={
        'UID': 'str',
        'CID': 'uint16',
        'Rating': 'uint8',
        'Timestamp': 'uint64'
      }
    )
    ratings['ContentType'] = 'MOVIELENS'
    return ratings