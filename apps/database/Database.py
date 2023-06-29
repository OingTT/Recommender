import os
import sqlalchemy

import pandas as pd

from dotenv import load_dotenv
from sqlalchemy.sql.dml import Update
from sqlmodel.engine.result import ScalarResult
from sqlmodel.orm.session import SelectOfScalar
from sqlmodel import (
  SQLModel,
  create_engine,
  Session,
  select,
  update,
)

from apps.database.models import *

class Database:
  def __init__(self, CFG: dict):
    load_dotenv()

    if CFG.get('debug'):
      DB_URL = os.getenv("WHATSUBS_TEST_DB_URL")
      ECHO = True
      # DB_URL = sqlalchemy.engine.URL.create(
      #   drivername='mysql',
      #   username=os.getenv("WHATSUBS_TEST_DB_USERNAME"),
      #   password=os.getenv("WHATSUBS_TEST_DB_PASSWORD"),
      #   host=os.getenv("WHATSUBS_TEST_DB_HOST"),
      #   port=os.getenv("WHATSUBS_TEST_DB_PORT"),
      #   database=os.getenv("WHATSUBS_TEST_DB_DATABASE")
      # )
    else:
      DB_URL = os.getenv("WHATSUBS_DB_URL")
      ECHO = False
      # DB_URL = sqlalchemy.engine.URL.create(
      #   drivername='mysql',
      #   username=os.getenv("WHATSUBS_DB_USERNAME"),
      #   password=os.getenv("WHATSUBS_DB_PASSWORD"),
      #   host=os.getenv("WHATSUBS_DB_HOST"),
      #   database=os.getenv("WHATSUBS_DB_DATABASE")
      # )   
    
    self.engine = create_engine(DB_URL, echo=ECHO)

    SQLModel.metadata.reflect(self.engine)

  def __getAllRecords(self, table: SQLModel) -> list[SQLModel]:
    '''
    Select all records from table\n
    usage: __getAllRecords(SQLModel)\n
    ex) __getAllRecords(SQLModel) -> [SQLModel(id='1', ...), SQLModel(id='2', ...), ...]
    '''
    with Session(self.engine) as session:
      statement = select(table)
      results: ScalarResult = session.exec(statement)
      return results.fetchall()

  def __findRecordByFields(self, table: SQLModel, **kwargs) -> SQLModel:
    '''
    Find single row by specified field and value pairs\n
    usage: __findRecordByFields(SQLModel, FIELD_NAME=FIELD_VALUE)\n
    ex) __findRecordByFields(SQLModel, id='1', field_name=field_value) -> SQLModel(id='1', field=field_value, ...)
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      for field_name, field_value in kwargs.items():
        statement = statement.where(table.__getattribute__(table, field_name) == field_value)
      results = session.exec(statement)
      return results.first()
    
  def __findRecordsByFields(self, table: SQLModel, **kwargs) -> list[SQLModel]:
    '''
    Find multiple row by specified field and value pairs\n
    usage: __findRecordsByFields(SQLModel, FIELD_NAME=FIELD_VALUE)\n
    ex) __findRecordsByFields(SQLModel, id='1', field_name=field_value) -> [SQLModel(id='1', field=field_value, ...), ...]
    '''
    with Session(self.engine) as session:
      statement: SelectOfScalar = select(table)
      for field_name, field_value in kwargs.items():
        statement = statement.where(table.__getattribute__(table, field_name) == field_value)
      results = session.exec(statement)
      return results.fetchall()
    
  def __insertRecord(self, record: SQLModel) -> None:
    '''
    Insert single record to table\n
    usage: __insertRecord(SQLModel)\n
    ex) __insertRecord(SQLModel(id='1', ...))
    '''
    with Session(self.engine) as session:
      session.add(record)
      session.commit()

  def __insertManyRecords(self, records: list[SQLModel]) -> None:
    '''
    Insert multiple records to table\n
    usage: __insertManyRecords([SQLModel, SQLModel, ...])\n
    ex) __insertManyRecords([SQLModel(id='1', ...), SQLModel(id='2', ...), ...])
    '''
    with Session(self.engine) as session:
      for record in records:
        session.add(record)
      session.commit()

  def __updateRecordByField(self, record: SQLModel, **kargs) -> None:
    '''
    Update single record to table\n
    usage: __updateRecordByFields(SQLModel)\n
    ex) __updateRecordByFields(SQLModel(id='1', ...))
    '''
    assert len(kargs) == 1, 'Only one field can be specified'
    with Session(self.engine) as session:
      statement: Update = update(record.__class__)
      for field_name, field_value in record.__dict__.items():
        if field_name == '_sa_instance_state':
          continue
        statement = statement.values({field_name: field_value})
      for field_name, field_value in kargs.items():
        statement = statement.where(record.__getattribute__(field_name) == field_value)
      print(statement)
      session.execute(statement)
      session.commit()
  
  def getAllUsers(self) -> list[User]:
    return self.__getAllRecords(User)

  def getAllReviews(self) -> list[Review]:
    return self.__getAllRecords(Review)
  
  def getAllUsersSubscribe(self) -> list[_SubscriptionToUser]:
    return self.__getAllRecords(_SubscriptionToUser)
  
  def getAllOTT(self) -> list[Subscription]:
    return self.__getAllRecords(Subscription)
  
  def __findRecordByField(self, table: SQLModel, field_name: str, field_value: str) -> SQLModel:
    return self.__findRecordByFields(table, **{field_name: field_value})
  
  def __findRecordsByField(self, table: SQLModel, field_name: str, field_value: str) -> list[SQLModel]:
    return self.__findRecordsByFields(table, **{field_name: field_value})

  def findUserByField(self, field_name: str, field_value: str) -> User:
    '''
    Find user by specified field and value\n
    usage: findUserByField(FIELD_NAME, FIELD_VALUE)\n
    ex) findUserByField('id', '1') -> User(id='1', ...)
    '''
    return self.__findRecordByField(User, field_name, field_value)
    
  def findReviewsByField(self, field_name: str, field_value: str) -> list[Review]:
    '''
    Find reviews by specified field and value\n
    usage: findReviewsByField(FIELD_NAME, FIELD_VALUE)\n
    ex) findReviewsByField('userId', '1') -> [Review(id='1', ...), Review(id='1', ...), ...]
    '''
    return self.__findRecordsByField(Review, field_name, field_value)

  def insertManyUserClustered(self, userClustered: list[UserClustered]) -> None:
    '''
    Insert single userClustered to table\n
    usage: insertUserClustered(USER_ID, CLUSTERED_ID)\n
    ex) insertUserClustered(id='1', ...)
    '''
    self.__insertManyRecords(userClustered)

  def updateUserClusteredByUserId(self, userClustered: UserClustered) -> None:
    '''
    Update single userClustered to table\n
    usage: updateUserClustered(USER_ID, CLUSTERED_ID)\n
    ex) updateUserClustered(id='1', ...)
    '''
    self.__updateRecordByField(userClustered, id=userClustered.id)
