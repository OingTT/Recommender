import os

import pandas as pd
import numpy as np

class movies(object):
  def __init__(self, csvPath) -> None:
    self.data = pd.read_csv(csvPath)

  def getTitleById(self, movieId) -> str:
    return self.data.loc[self.data['movieId'] == movieId]['title']

  def getGenresById(self, movieId) -> list:
    return self.data.loc[self.data['movieId'] == movieId]['genres']
  
  def getIdByTitle(self, title):
    return self.data.loc[self.data['title'] == title]['movieId']
  
  def getIdByTitle(self, title):
    return self.data.loc[self.data['title'] == title]['genres']

class MovieLens(object):
  def __init__(self, movieLenslDir):
    self.baseDir = os.path.abspath(movieLenslDir)
    self._loadMLRating()
    self._createUserItemMatrix()
  
  def _deleteNewLineFromSplittedString(self, strings: list):
    if strings[-1][-1] == '\n':
      strings[-1] = strings[-1][:-1]
    return strings

  def _loadMLRating(self):
    '''
    ml-1m
    '''
    userIds, movieIds, ratings, timestamps = [[] for _ in range(4)]
    with open(f'{self.baseDir}\\ratings.dat') as f:
      for line in f.readlines():
        splitted = line.split('::')
        splitted = self._deleteNewLineFromSplittedString(splitted)
        userIds.append(splitted[0])
        movieIds.append(splitted[1])
        ratings.append(splitted[2])
        timestamps.append(splitted[3])
    self.Ratings = pd.DataFrame({
      'UserId': userIds,
      'MovieId': movieIds,
      'Rating': ratings,
      'Timestamp': timestamps
    }).astype({
      'UserId': 'uint16',
      'MovieId': 'uint16',
      'Rating': 'uint8',
      'Timestamp': 'string'
    })
    
  def _createUserItemMatrix(self):
    self.UserItemMatrix = self.Ratings.pivot_table(index='UserId', columns='MovieId', values='Rating')
    # print(f"Count NAN: {user_item_matrix.isna().sum().sum()}")
    self.UserItemMatrix = self.UserItemMatrix.fillna(0)

  def _loadUserDemographicInfo(self):
    userIds, userGenders, userAges, userOccupations, userZips = [[] for _ in range(5)]
    with open(f'{self.baseDir}\\users.dat') as f:
      for line in f.readlines():
        splitted = line.split('::')
        splitted = self._deleteNewLineFromSplittedString(splitted)
        userIds.append(splitted[0])
        userGenders.append(splitted[1])
        userAges.append(splitted[2])
        userOccupations.append(splitted[3])
        userZips.append(splitted[4])
    self.UserDemographicInfo = pd.DataFrame({
      'UserId': userIds,
      'Gender': userGenders,
      'Age': userAges,
      'Occupation': userOccupations,
      'ZIP-Code': userZips
    }).astype({
      'UserId': 'uint16',
      'Gender': 'str',
      'Age': 'uint8',
      'Occupation': 'uint8',
      'ZIP-Code': 'string'
    })

m = MovieLens('ml-1m')
m._loadUserDemographicInfo()