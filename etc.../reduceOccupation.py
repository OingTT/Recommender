import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

if not os.path.exists('ml-1m/meanByOccupation.csv') or not os.path.exists('ml-1m/sumByOccupation.csv'):
  users_df = pd.read_csv('ml-1m/users.dat', sep='::', engine='python',
                          names=['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code'])
  ratings_df = pd.read_csv('ml-1m/ratings.dat', sep='::', engine='python',
                          names=['UserID', 'MovieID', 'Rating', 'Timestamp'])

  UIMatrix: pd.DataFrame = ratings_df.pivot_table(index='UserID', columns='MovieID', values='Rating').fillna(0)

  sumByOccupation: dict = {_: None for _ in range(0, 21)}

  for row in UIMatrix.iterrows():
    userID: int = row[0]
    movieRatings: pd.Series = row[1]
    occu: int = users_df[users_df['UserID'] == row[0]]['Occupation'].values[0]
    if sumByOccupation[occu] is not None:
      sumByOccupation[occu]: pd.Series = sumByOccupation[occu].add(movieRatings, fill_value=0)
    else:
      sumByOccupation[occu]: pd.Series = movieRatings

  meanByOccupation = {_: None for _ in range(0, 21)}
  for k, v in sumByOccupation.items():
    meanByOccupation[k] = sumByOccupation[k] / users_df[users_df['Occupation'] == k].shape[0]

  sumByOccupation = pd.DataFrame(sumByOccupation)
  sumByOccupation.to_csv('ml-1m/sumByOccupation.csv', index=False)

  meanByOccupation = pd.DataFrame(meanByOccupation)
  meanByOccupation.to_csv('ml-1m/meanByOccupation.csv', index=False)
else:
  sumByOccupation = pd.read_csv('ml-1m/sumByOccupation.csv')
  meanByOccupation = pd.read_csv('ml-1m/meanByOccupation.csv')

corr = meanByOccupation.corr()

### 대각 성분 0으로 채우기
np.fill_diagonal(corr.values, 0)

couple = dict()
for i in range(1, len(corr.columns)):
  # 상관계수가 가장 높은 순으로 정렬
  max_ = corr.iloc[i].nlargest(20)
  for j in range(20):
    high_corr = corr.iloc[i][corr.iloc[i] == max_[j]].index[0]
    if str(i) not in couple.keys() \
      and str(i) not in couple.values() \
        and high_corr not in couple.keys() \
          and high_corr not in couple.values()\
            and int(high_corr) != 0: # 0: other은 제외하도록
      couple[corr.columns[i]] = high_corr
      break

print(couple)

np.fill_diagonal(corr.values, 1)
sns.heatmap(corr, 
               annot = True,      # 실제 값 화면에 나타내기
               cmap = 'RdYlBu_r',  # Red, Yellow, Blue 색상으로 표시
               vmin = -1, vmax = 1, #컬러차트 -1 ~ 1 범위로 표시
              )
plt.show()

### Reduced by correalation
'''
0 + 14  : Other/Sales/Marketing
1 + 6   : Academic/Educator/Doctor/Health Care
2 + 20  : Artist/Writer
3 + 7   : Clerical/Admin/Executives/Managerial
4 + 12  : College/Grad Student/Programmer
5 + 17  : Customer Service/Technician/Engineer
8 + 15  : Farmer/Scientist
9 + 16  : Homemaker/Self-Employed
10 + 19 : K-12 Student/Unemployed
13 + 11 : Retired/Lawyer
'''
### Reduced by correalation except 0
'''
1 + 6   : Academic/Educator/Doctor/Health Care
2 + 20  : Artist/Writer
3 + 7   : Clerical/Admin/Executives/Managerial
4 + 14  : College/Grad Student/Sales/Marketing
5 + 17  : Customer Service/Technician/Engineer
8 + 12  : Farmer/Programmer
9 + 16  : Homemaker/Self-Employed
10 + 19 : K-12 Student/Unemployed
11 + 15 : Lawyer/Scientist
13 + 18 : Retired/Tradesman/Craftsman
'''
### Reduced by rating count
'''
0: other                      2093
4: college/grad student       759
7: execytive/managerial       679
1: academic/educator          528
17: technician/engineer       502
12: programmer                388
14: sales/marketing           302
20: writer                    281
2: artist                     267
16: self-employed             241
'''

'''
	*  0:  "other" or not specified
	*  1:  "academic/educator"
	*  2:  "artist"
	*  3:  "clerical/admin"
	*  4:  "college/grad student"
	*  5:  "customer service"
	*  6:  "doctor/health care"
	*  7:  "executive/managerial"
	*  8:  "farmer"
	*  9:  "homemaker"
	* 10:  "K-12 student"
	* 11:  "lawyer"
	* 12:  "programmer"
	* 13:  "retired"
	* 14:  "sales/marketing"
	* 15:  "scientist"
	* 16:  "self-employed"
	* 17:  "technician/engineer"
	* 18:  "tradesman/craftsman"
	* 19:  "unemployed"
	* 20:  "writer"
'''