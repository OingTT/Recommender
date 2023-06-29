import datetime

from apps.database.models import *
from apps.database.Database import Database
from apps.database.DatabaseAdapter import DatabaseAdapter
from apps.arg_parser import get_args

cfgs = get_args()
cfgs['debug'] = True

# db = Database(cfgs)

user = User(
    id='clhkairuv0000mn08gt2yvi5b',
    email='barrow13512@gmail.com',
    image='https://lh3.googleusercontent.com/a/AGNmyxYyeHbbdRvUVTRoyLLHo8KCZ46ITbk2T0K0Y7gU=s96-c',
    birth=datetime.datetime(1999, 8, 1, 0, 0),
    gender='M',
    emailVerified=None,
    name='김민재',
    avatar='https://lh3.googleusercontent.com/a/AGNmyxYyeHbbdRvUVTRoyLLHo8KCZ46ITbk2T0K0Y7gU=s96-c',
    occupationId=Occupation(id=4, name='학생')
  )

dba = DatabaseAdapter(cfgs)
dba.updateUserClusteredByUserId(id='clhkairuv0000mn08gt2yvi5b', label=3)