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
    return users
  
  def getAllReviews(self) -> DataFrame:
    reviews = self.recordsToDataFrame(self.database.getAllReviews())
    return reviews[reviews['watch'] == WatchStatus.WATCHED]
  
  def getReviewByContentType(self, contentType: str) -> DataFrame:
    reviews = self.recordsToDataFrame(self.database.findReviewsByField(field_name='contentType', field_value=contentType))
    return reviews[reviews['watch'] == WatchStatus.WATCHED]
  
  def getAllUsersSubscribe(self) -> DataFrame:
    user_subscribe = self.recordsToDataFrame(self.database.getAllUsersSubscribe())
    user_subscribe = user_subscribe.rename(columns={'A': 'Subscription', 'B': 'UID'})
    return user_subscribe
  
  def getAllOTT(self) -> DataFrame:
    return self.recordsToDataFrame(self.database.getAllOTT())
  
  def getAllUserClustered(self) -> DataFrame:
    return self.recordsToDataFrame(self.database.getAllUserClustered())
  
  def insertUserClustered(self, userId: str, clusterId: int) -> None:
    self.database.insertUserClustered(userId=userId, clusterId=clusterId)

  def insertManyUserClustered(self, user_clustered: DataFrame) -> None:
    self.database.insertManyUserClustered(user_clustered=user_clustered)

  def updateUserClusteredByUserId(self, userId: str, label: int) -> None:
    new_user_clustered = UserClustered(id=userId, label=label)
    self.database.updateUserClusteredByUserId(new_user_clustered)