import os

import pandas as pd

from dotenv import load_dotenv
from sqlmodel.engine.result import ScalarResult
from sqlmodel.orm.session import SelectOfScalar
from sqlmodel import (
  SQLModel,
  create_engine,
  Session,
  select
)

from models import *

class Database:
  def __init__(self):
    load_dotenv()
    DB_URL = os.getenv("WHATSUBS_DB_URL")

    self.engine = create_engine(DB_URL)

    SQLModel.metadata.reflect(self.engine)

  def recordsToDataFrame(self, records: list[SQLModel]) -> pd.DataFrame:
    return pd.DataFrame([record.dict() for record in records])

  def __getAllRecords(self, table: SQLModel) -> list[SQLModel]:
    with Session(self.engine) as session:
      statement = select(table)
      results: ScalarResult = session.exec(statement)
      return results.fetchall()

  def getAllUsers(self) -> list[User]:
    return self.__getAllRecords(User)

  def getAllReviews(self) -> list[Review]:
    return self.__getAllRecords(Review)
  
  def findRecordByFields(self, table: SQLModel, **kwargs) -> SQLModel:
    '''
    Find row by specified field and value pairs\n
    usage: __findRecordByFields(SQLModel, FIELD_NAME=FIELD_VALUE)\n
    ex) __findRecordByFields(SQLModel, id='1', field=field_value) -> SQLModel(id='1', field=field_value, ...)
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      for key, value in kwargs.items():
        statement = statement.where(table.__getattribute__(table, key) == value)
      results = session.exec(statement)
      return results.first()
    
  def findRecordsByFields(self, table: SQLModel, **kwargs) -> SQLModel:
    '''
    Find row by specified field and value pairs\n
    usage: __findRecordByFields(SQLModel, FIELD_NAME=FIELD_VALUE)\n
    ex) __findRecordByFields(SQLModel, id='1', field=field_value) -> SQLModel(id='1', field=field_value, ...)
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      for key, value in kwargs.items():
        statement = statement.where(table.__getattribute__(table, key) == value)
      results = session.exec(statement)
      return results.fetchall()
  
  def __findRecordByField(self, table: SQLModel, field: str, field_value: str) -> SQLModel:
    '''
    Find row by specified field and value\n
    usage: __findRecordByField(SQLModel, FIELD_NAME, FIELD_VALUE)\n
    ex) __findRecordByField(SQLModel, 'id', '1') -> SQLModel(id='1', ...)
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      statement = statement.where(table.__getattribute__(table, field) == field_value)
      results = session.exec(statement)
      return results.first()
  
  def __findRecordsByField(self, table: SQLModel, field: str, field_value: str) -> list[SQLModel]:
    '''
    Find row by specified field and value\n
    usage: __findRecordsByField(SQLModel, FIELD_NAME, FIELD_VALUE)\n
    ex) __findRecordsByField(SQLModel, 'id', '1') -> [SQLModel(id='1', ...), SQLModel(id='1', ...), ...]
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      statement = statement.where(table.__getattribute__(table, field) == field_value)
      results = session.exec(statement)
      return results.fetchall()

  def findUserByField(self, field: str, field_value: str) -> User:
    '''
    Find user by specified field and value\n
    usage: findUserByField(FIELD_NAME, FIELD_VALUE)\n
    ex) findUserByField('id', '1') -> User(id='1', ...)
    '''
    return self.__findRecordByField(User, field, field_value)
    
  def findReviewsByField(self, field: str, field_value: str) -> list[Review]:
    '''
    Find reviews by specified field and value\n
    usage: findReviewsByField(FIELD_NAME, FIELD_VALUE)\n
    ex) findReviewsByField('userId', '1') -> [Review(id='1', ...), Review(id='1', ...), ...]
    '''
    return self.__findRecordsByField(Review, field, field_value)
