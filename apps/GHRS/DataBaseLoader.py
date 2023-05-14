import os
import mysql.connector

from __cert__ import __cert__

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
      self.HOST = __cert__.get('WHATSUBS_HOST')
      self.USERNAME = __cert__.get('WHATSUBS_USERNAME')
      self.PASSWORD = __cert__.get('WHATSUBS_PASSWORD')
      self.DATABASE = __cert__.get('WHATSUBS_DATABASE')
      self.connection = None
      self.cursor = None

  def __connect__(self, ) -> None:
    self.connection = mysql.connector.connect(
      host=self.HOST,
      database=self.DATABASE,
      user=self.USERNAME,
      password=self.PASSWORD,
      ssl_ca=os.getenv("SSL_CERT")
    )
    self.cursor = self.connection.cursor()

  def __disconnect__(self, ) -> None:
    if self.cursor is not None:
      self.cursor.close()
    if self.connection is not None and self.connection.is_connected():
      self.connection.close()

  def execute(self, query: str):
    if self.connection is None or self.connection.is_connected() is False:
      self.__connect__()
    self.cursor.execute(query)
    return self.cursor.fetchall()

dbLoader = DataBaseLoader.getInstance()
print(dbLoader.execute('show tables'))
print(dbLoader.execute('SELECT * FROM Review'))