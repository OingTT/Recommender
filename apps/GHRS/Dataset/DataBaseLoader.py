import os
import mysql.connector
import traceback

import pandas as pd

from datetime import datetime
from dotenv import load_dotenv

from apps.Singleton import Singleton
from apps.apis.TMDB import TMDB

class DataBaseLoader(metaclass=Singleton):

  def __init__(self, ):
    load_dotenv()
    self.HOST = os.getenv("WHATSUBS_DB_HOST")
    self.USERNAME = os.getenv("WHATSUBS_DB_USERNAME")
    self.PASSWORD = os.getenv("WHATSUBS_DB_PASSWORD")
    self.DATABASE = os.getenv("WHATSUBS_DB_DATABASE")
    self.connection = None
    self.cursor = None
    self.TMDB_API = TMDB()

  def __connect(self, ) -> None:
    self.connection = mysql.connector.connect(
      host=self.HOST,
      database=self.DATABASE,
      user=self.USERNAME,
      password=self.PASSWORD,
      ssl_ca=os.getenv("SSL_CERT")
    )

  def __disconnect(self, ) -> None:
    if self.cursor is not None:
      self.cursor.close()
    if self.connection is not None and self.connection.is_connected():
      self.connection.close()

  def __execute(self, query: str, *args: tuple) -> list:
    fail_cnt = 0
    while True:
      if fail_cnt >= 3:
        print('Fail count is over 3. Stop to execute query: {}'.format(query))
        result = None
      try:
        if self.connection is None or self.connection.is_connected() is False:
          self.__connect()
        args = (*args, )          
        cursor = self.connection.cursor()
        if len(args) != 0:
          cursor.execute(query, args)
        else:
          cursor.execute(query)
        break
      except mysql.connector.InterfaceError as e:
        result = None
        break
      except Exception as e:
        print('Fail to execute query: {}'.format(query))
        print(e)
        traceback.print_exc()
        fail_cnt += 1
        continue
      finally:
        if cursor is not None:
          result = cursor.fetchall()
          cursor.close()
        else:
          result = None
    return result
  
  def __getAge(self, birthday: datetime) -> int:
    if birthday is None:
      return None
    today = datetime.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
  
  def getReviewsByContentType(self, contentType: str) -> pd.DataFrame:
    query = 'SELECT * FROM Review WHERE CONTENT_TYPE = %s'
    response = self.__execute(query, contentType)
    reviews = list()
    for row in response:
      if row[3] != 'WATCHED':
        continue
      UID = row[0]
      CONTENT_TYPE = row[1]
      CONTENT_ID = row[2]
      WATCH = row[3]
      RATING = row[4]
      reviews.append({
        'UID': UID,
        'CID': CONTENT_ID,
        'Rating': RATING,
        'ContentType': CONTENT_TYPE,
        'Timestamp': None,
      })
    return pd.DataFrame().from_records(reviews)

  def getAllReviews(self) -> pd.DataFrame:
    query = 'SELECT * FROM Review'
    response = self.__execute(query)
    reviews = list()
    for row in response:
      if row[3] != 'WATCHED':
        continue
      UID = row[0]
      CONTENT_TYPE = row[1]
      CONTENT_ID = row[2]
      WATCH = row[3]
      RATING = row[4]
      reviews.append({
        'UID': UID,
        'CID': CONTENT_ID,
        'Rating': RATING,
        'ContentType': CONTENT_TYPE,
        'Timestamp': None,
      })
    return pd.DataFrame().from_records(reviews)

  def getAllUsers(self, ) -> pd.DataFrame:
    query = 'SELECT * FROM User'
    response = self.__execute(query)
    users = list()
    for row in response:
      UID = row[0]
      NAME = row[1]
      EMAIL = row[2]
      EMAIL_VERIFIED = row[3]
      PROFILE_IMAGE = row[4]
      PROFILE_AVATAR = row[5]
      BIRTHDAY = row[6]
      OCCUPATION = row[7]
      GENDER = row[8]
      if UID is None or BIRTHDAY is None or OCCUPATION is None or GENDER is None: # 필수 정보가 없는 경우 제외
        continue
      AGE = self.__getAge(BIRTHDAY)
      if AGE <= 0:
        continue
      users.append({
        'UID': UID,
        'Gender': GENDER,
        'Age': AGE,
        'Occupation': OCCUPATION,
        'Zip': None,
      })
    return pd.DataFrame().from_records(users)
  
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
