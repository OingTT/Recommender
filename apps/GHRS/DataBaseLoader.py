import os
import mysql.connector

import pandas as pd

from datetime import datetime
from dotenv import load_dotenv

class DataBaseLoader():
  __instance__ = None

  @staticmethod
  def getInstance() -> 'DataBaseLoader':
    if DataBaseLoader.__instance__ is None:
      return DataBaseLoader()
    return DataBaseLoader.__instance__

  def __init__(self, ):
    if self.__instance__ is not None:
      raise Exception("Singleton class, use get_instance() method instead")
    else:
      load_dotenv()
      self.HOST = os.getenv("WHATSUBS_DB_HOST")
      self.USERNAME = os.getenv("WHATSUBS_DB_USERNAME")
      self.PASSWORD = os.getenv("WHATSUBS_DB_PASSWORD")
      self.DATABASE = os.getenv("WHATSUBS_DB_DATABASE")
      self.connection = None
      self.cursor = None

  def __connect(self, ) -> None:
    self.connection = mysql.connector.connect(
      host=self.HOST,
      database=self.DATABASE,
      user=self.USERNAME,
      password=self.PASSWORD,
      ssl_ca=os.getenv("SSL_CERT")
    )
    self.cursor = self.connection.cursor()

  def __disconnect(self, ) -> None:
    if self.cursor is not None:
      self.cursor.close()
    if self.connection is not None and self.connection.is_connected():
      self.connection.close()

  def __execute(self, query: str):
    if self.connection is None or self.connection.is_connected() is False:
      self.__connect()
    self.cursor.execute(query)
    return self.cursor.fetchall()
  
  def __getAge(self, birthday: datetime) -> int:
    '''
    Movie Lens는 1998년에 만들어진 데이터셋이므로, 1998년을 기준으로 나이를 계산했을듯
    '''
    today = datetime.today()
    return today.year - birthday.year - ((today.month, today.day) < (birthday.month, birthday.day))
  
  def getReviews(self, ) -> pd.DataFrame:
    '''
    TODO 
    Movie Lens는 IMDB ID를 줌
    그러나 WhatSubs는 TMDB ID를 줌
    따라서, TMDB ID를 IMDB ID로 변환하는 작업이 필요함
    '''
    query = 'SELECT * FROM Review'
    response = self.__execute(query)
    reviews = list()
    for row in response:
      UID = row[0]
      CONTENT_TYPE = row[1]
      CONTENT_ID = row[2]
      WATCH = row[3]
      RATING = row[4]
      if WATCH != 'WATCHED':
        continue
      if CONTENT_TYPE == 'MOVIE':
        '''
        Movie Lens에는 TV Show에 대한 리뷰가 없으므로
        TV Show에 대한 사용자 평가가 충분히 생기기 전 까지는
        일단 영화에 대한 추천만 할 예정이므로
        영화에 대한 Review만 추출
        '''
        MID = CONTENT_ID
        reviews.append({
          'UID': UID,
          'MID': MID,
          'Rating': RATING,
          'Timestamp': None,
        })
    return pd.DataFrame().from_records(reviews)


  def getUsers(self, ) -> pd.DataFrame:
    query = 'SELECT * FROM User'
    response = self.__execute(query)
    users = list()
    for row in response:
      UID = row[0]
      NAME = row[1]
      EMAIL = row[2]
      EMAIL_VERIFIED = row[3] # 뭔지 모르겠음
      PROFILE_IMAGE = row[4]
      PROFILE_AVATAR = row[5]
      BIRTHDAY = row[6]
      OCCUPATION = row[7]
      GENDER = row[8]
      AGE = self.__getAge(BIRTHDAY)
      users.append({
        'UID': UID,
        'Gender': GENDER,
        'Age': AGE,
        'Occupation': OCCUPATION,
        'Zip': None,
      })
    return pd.DataFrame().from_records(users)
  
  def ___(self, ):
    return self.__execute('show columns from Review')
  
from pprint import pprint

dbLoader = DataBaseLoader.getInstance()
pprint(dbLoader.___())
