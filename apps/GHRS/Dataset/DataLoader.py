import pandas as pd

from abc import abstractmethod

class DataLoader:
  def __init__(self) -> None:
    pass
  
  @abstractmethod
  def getAllUsers(self) -> pd.DataFrame:
    '''
    Columns: UID, Gender, Age, Occupation, Zip
    '''
    ...

  @abstractmethod
  def getAllReviews(self) -> pd.DataFrame:
    '''
    Columns: UID, CID, Rating, ContentType, Timestamp
    '''
    ...

  @abstractmethod
  def getReviewsByContentType(self, contentType: str) -> pd.DataFrame:
    '''
    Columns: UID, CID, Rating, ContentType, Timestamp
    '''
    ...