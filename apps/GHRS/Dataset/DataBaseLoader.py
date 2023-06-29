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
    users = self.databaseAdapter.getAllUsers()
    return users[['id', 'gender', 'age', 'occupationId']]
  
  def getAllUserSubscribe(self, ) -> pd.DataFrame:
    return self.databaseAdapter.getAllUsersSubscribe()
  
  def getAllUserClustered(self):
    return self.databaseAdapter.getAllUserClustered()
  
  def getAllOTT(self):
    return self.databaseAdapter.getAllOTT()
  
  def getUserClusteredByUserId(self, id: str) -> pd.DataFrame:
    return self.databaseAdapter.getUserClusteredByUserId(id=id)
  
  def getUserClusteredByClusterLabel(self, cluster_label: int) -> pd.DataFrame:
    return self.databaseAdapter.getUserClusteredByClusterLabel(cluster_label=cluster_label)