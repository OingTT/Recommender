from apps.GHRS.DataBaseLoader import DataBaseLoader

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
# len_reviews_by_name(dbLoader, '박준서')
# len_reviews_by_name(dbLoader, 'dong')
# len_reviews_by_name(dbLoader, '이정준')
# len_reviews_by_name(dbLoader, '김종인')
# len_reviews_by_email(dbLoader, 'dalek76@naver.com')
# len_reviews_by_name(dbLoader, '은댕')

users = dbLoader._test('SELECT * FROM User')
for user in users:
  if user[6] == None:
    print("!!!!!")
