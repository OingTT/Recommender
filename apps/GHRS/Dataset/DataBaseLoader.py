import pandas as pd

from apps.apis.TMDB import TMDB
from apps.utils.Singleton import Singleton
from apps.database.DatabaseAdapter import DatabaseAdapter
from apps.GHRS.Dataset.DataLoader import DataLoader

class DataBaseLoader(DataLoader, metaclass=Singleton):

  def __init__(self, databaseAdapter: DatabaseAdapter=None) -> None:
    if databaseAdapter is None:
      databaseAdapter = DatabaseAdapter()
    self.databaseAdapter = databaseAdapter
    self.TMDB_API = TMDB()
  
  def getReviewsByContentType(self, contentType: str) -> pd.DataFrame:
    return self.databaseAdapter.getReviewByContentType(contentType=contentType)

  def getAllReviews(self) -> pd.DataFrame:
    return self.databaseAdapter.getAllReviews()

  def getAllUsers(self) -> pd.DataFrame:
    return self.databaseAdapter.getAllUsers()
  
  def getAllUserSubscribe(self, ) -> pd.DataFrame:
    return self.databaseAdapter.getAllUsersSubscribe()
  
  def getAllOTT(self):
    return self.databaseAdapter.getAllOTT()