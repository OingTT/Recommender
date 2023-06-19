import os
import mysql.connector
import traceback

import pandas as pd

import apps.database.models as Models

from datetime import datetime
from dotenv import load_dotenv

from apps.apis.TMDB import TMDB
from apps.Singleton import Singleton
from apps.database.database import Database
from apps.GHRS.Dataset.DataLoader import DataLoader

class DataBaseLoader(DataLoader, metaclass=Singleton):

  def __init__(self):
    self.database = Database()
    self.TMDB_API = TMDB()
  
  def __getAge(self, birthday: datetime) -> int:
    if birthday is pd.NaT or birthday is None:
      return None
    today = datetime.today()
    return int(today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day)))
  
  def __birthToAge(self, df: pd.DataFrame) -> int:
    return self.__getAge(df['birth'])
  
  def getReviewsByContentType(self, contentType: str) -> pd.DataFrame:
    records = self.database.findRecordsByFields(table=Models.Review, CONTENT_TYPE=contentType, WATCH='WATCHED')
    return self.database.recordsToDataFrame(records)

  def getAllReviews(self) -> pd.DataFrame:
    records = self.database.findRecordsByFields(table=Models.Review, WATCH='WATCHED')
    return self.database.recordsToDataFrame(records)

  def getAllUsers(self, ) -> pd.DataFrame:  
    records = self.database.findRecordsByFields(table=Models.User)
    records = self.database.recordsToDataFrame(records)
    records['age'] = records.apply(self.__birthToAge, axis=1)
    records = records.drop(columns=['birth'], axis=1)
    # id, gender, occupationId, Age가 결측치인 행 제거
    return records.dropna(how='any', subset=['id', 'gender', 'occupationId',  'age'])
  
  def getAllUserSubscribe(self, ) -> pd.DataFrame:
    query = 'SELECT * FROM _SubscriptionToUser'
    response = self.__execute(query)
    subscriptions = list()
    for row in response:
      UID = row[1]
      SUBSCRIPTION = row[0]
      subscriptions.append({
        'UID': UID,
        'Subscription': SUBSCRIPTION,
      })
    return pd.DataFrame().from_records(subscriptions)
  
  def getAllOTT(self):
    query = 'SELECT * FROM Subscription'
    response = self.__execute(query)
    subscriptions = list()
    for row in response:
      OID = row[0]
      ENGLISH_NAME = row[1]
      KOREAN_NAME = row[2]
      PROVIDER_ID = row[3]
      NETWORK_ID = row[4]
      PRICE = row[5]
      SHARING = row[6]
      subscriptions.append({
        'OID': OID,
        'E_NAME': ENGLISH_NAME,
        'K_NAME': KOREAN_NAME,
        'PROVIDER_ID': PROVIDER_ID,
        'NETWORK_ID': NETWORK_ID,
        'PRICE': PRICE,
        'SHARING': SHARING
      })
    return pd.DataFrame().from_records(subscriptions)
  
  def findUserByUserId(self, uid: str) -> dict:
    return self.__execute('SELECT * FROM User WHERE id=%s;', uid)
  
  def findUserByUserName(self, name: str) -> dict:
    return self.__execute('SELECT * FROM User WHERE name=%s;', name)
  
  def findUserByUserEmail(self, email: str) -> dict:
    return self.__execute('SELECT * FROM User WHERE email=%s;', email)

  def findReviesByUserId(self, uid: str) -> list:
    return self.__execute('SELECT * FROM Review WHERE userId=%s;', uid)
  
  def _test(self, args):
    return self.__execute(args)

dbl = DataBaseLoader()

print(dbl.getAllUsers())