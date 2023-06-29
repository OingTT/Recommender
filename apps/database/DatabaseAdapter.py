import pandas as pd

from pandas import DataFrame
from datetime import datetime
from sqlmodel import SQLModel

from apps.database.models import *
from apps.database.Database import Database

class DatabaseAdapter:
  def __init__(self, CFG: dict) -> None:
    self.database = Database(CFG=CFG)

  def __birthToAge(self, birthday: datetime) -> int:
    if birthday is pd.NaT or birthday is None:
      return None
    today = datetime.today()
    return int(today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day)))
  
  def recordsToDataFrame(self, records: list[SQLModel]) -> DataFrame:
    return DataFrame([record.dict() for record in records])

  def getAllUsers(self) -> DataFrame:
    users = self.recordsToDataFrame(self.database.getAllUsers())
    # birth, gender, name, id, occupationId가 결측치인 행 제거
    users = users.dropna(axis=0, subset=['birth', 'gender', 'name', 'id', 'occupationId'])
    users['occupationId'] = users['occupationId'].astype('int')
    users['age'] = users.apply(lambda row: self.__birthToAge(row['birth']), axis=1).astype('int')
    users = users.dropna(axis=0, subset=['age'])
    users = users[users['age'] > 0]
    return users.reset_index(drop=True)
  
  def getAllReviews(self) -> DataFrame:
    reviews = self.recordsToDataFrame(self.database.getAllReviews())
    reviews = reviews.dropna(axis=0, subset=['userId', 'contentId', 'rating', 'contentType', 'watch'])
    return reviews[reviews['watch'] == WatchStatus.WATCHED].reset_index(drop=True)
  
  def getReviewByContentType(self, contentType: str) -> DataFrame:
    reviews = self.recordsToDataFrame(self.database.findReviewsByField(field_name='contentType', field_value=contentType))
    reviews = reviews.dropna(axis=0, subset=['userId', 'contentId', 'rating', 'contentType', 'watch'])
    return reviews[reviews['watch'] == WatchStatus.WATCHED].reset_index(drop=True)
  
  def getAllUsersSubscribe(self) -> DataFrame:
    user_subscribe = self.recordsToDataFrame(self.database.getAllUsersSubscribe())
    user_subscribe = user_subscribe.rename(columns={'A': 'Subscription', 'B': 'UID'})
    return user_subscribe
  
  def getAllOTT(self) -> DataFrame:
    return self.recordsToDataFrame(self.database.getAllOTT())
  
  def getAllUserClustered(self) -> DataFrame:
    return self.recordsToDataFrame(self.database.getAllUserClustered())
  
  def getUserClusteredByUserId(self, id: str) -> DataFrame:
    return self.recordsToDataFrame(self.database.findUserClusteredByField(field_name='id', field_value=id))
  
  def getUserClusteredByClusterLabel(self, cluster_label: int) -> DataFrame:
    return self.recordsToDataFrame(self.database.findUserClusteredByField(field_name='label', field_value=cluster_label))
  
  def insertManyUserClustered(self, user_clustered: DataFrame) -> None:
    self.database.insertManyUserClustered(user_clustered=user_clustered)

  def updateUserClusteredByUserId(self, id: str, label: int) -> None:
    new_user_clustered = UserClustered(id=id, label=label)
    try:
      self.database.updateUserClusteredByUserId(new_user_clustered)
    except ValueError as e:
      result = self.database.findUserByField(field_name='id', field_value=id)
      if result is None:
        raise ValueError(f'No such user with id={id}')
      else:
        print(e, 'at UserClustered. Inserting new row')
        self.database.insertUserClustered(new_user_clustered)
