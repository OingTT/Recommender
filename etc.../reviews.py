import pandas as pd

from apps.GHRS.Dataset.DataBaseLoader import DataBaseLoader

def getAllReviews(dbLoader: DataBaseLoader) -> list:
  reviews = dbLoader._test("SELECT * FROM Review")
  reviews_list = list()
  for row in reviews:
    UID = row[0]
    CONTENT_TYPE = row[1]
    CONTENT_ID = row[2]
    WATCH = row[3]
    RATING = row[4]

    if WATCH != 'WATCHED':
      continue
    
    reviews_list.append({
      'UID': UID,
      'CONTENT_TYPE': CONTENT_TYPE,
      'CONTENT_ID': CONTENT_ID,
      'WATCH': WATCH,
      'RATING': RATING,
    })
  return pd.DataFrame.from_records(reviews_list)

def getAllUsers(dbLoader: DataBaseLoader) -> list:
  users = dbLoader._test("SELECT * FROM User")
  users_list = list()
  for row in users:
    UID = row[0]
    NAME = row[1]
    EMAIL = row[2]
    EMAIL_VERIFIED = row[3]
    PROFILE_IMAGE = row[4]
    PROFILE_AVATAR = row[5]
    BIRTHDAY = row[6]
    OCCUPATION = row[7]
    GENDER = row[8]

    if BIRTHDAY is None or OCCUPATION is None or GENDER is None: # 필수 정보가 없는 경우 제외
      continue
    
    users_list.append({
      'UID': UID,
      'NAME': NAME,
      'EMAIL': EMAIL,
      'EMAIL_VERIFIED': EMAIL_VERIFIED,
      'PROFILE_IMAGE': PROFILE_IMAGE,
      'PROFILE_AVATAR': PROFILE_AVATAR,
      'BIRTHDAY': BIRTHDAY,
      'OCCUPATION': OCCUPATION,
      'GENDER': GENDER
    })
  return pd.DataFrame.from_records(users_list)

def len_reviews_by_name(dbLoader: DataBaseLoader, userName):
  cnt = 0
  user = dbLoader.findUserByUserName(userName)
  uid = user[0][0]
  reviews = dbLoader.findReviesByUserId(uid)
  for row in reviews:
    if row[3] != 'WATCHED':
      continue
    cnt += 1
  print(f'{userName}: {cnt}')
def len_reviews_by_email(dbLoader: DataBaseLoader, userEmail):
  cnt = 0
  user = dbLoader.findUserByUserEmail(userEmail)
  userName = user[0][1]
  uid = user[0][0]
  reviews = dbLoader.findReviesByUserId(uid)
  for row in reviews:
    if row[3] != 'WATCHED':
      continue
    cnt += 1
  print(f'{userName}: {cnt}')


dbLoader = DataBaseLoader()

print(dbLoader.getAllOTT())

# clhkairuv0000mn08gt2yvi5b
# print(dbLoader._test('SELECT id FROM User WHERE name="김민재"'))


# len_reviews_by_name(dbLoader, '박준서')
# len_reviews_by_name(dbLoader, '이정준')
# len_reviews_by_name(dbLoader, '김종인')
# len_reviews_by_email(dbLoader, 'dalek76@naver.com')
# len_reviews_by_name(dbLoader, '은댕')
# len_reviews_by_name(dbLoader, 'dong')
# len_reviews_by_name(dbLoader, '박다영')

# reviews = getAllReviews(dbLoader)
# print(f'# of all Review: {len(reviews)}')

# users = getAllUsers(dbLoader)
# print(f'# of all User: {len(users)}')

# cnt_review_each = list()

# for uid in users['UID']:
#   review_by = reviews[reviews['UID'] == uid]
#   cnt_review_each.append({
#     'UID': uid,
#     'CNT': len(review_by)
#   })

# cnt_review_each_df = pd.DataFrame().from_records(cnt_review_each, columns=['UID', 'CNT'])

# print(cnt_review_each_df.sort_values('CNT', ascending=False))