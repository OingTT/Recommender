import enum

from typing import Optional
from datetime import datetime
from sqlalchemy.ext.declarative import declared_attr

from sqlmodel import (
  Field,
  SQLModel,
  Enum,
  Column,
  Relationship,
)

__all__ = [
  'User',
  'Review',
  'Account',
  'CategoriesOnUsers',
  'Category',
  'Genre',
  'Occupation',
  'Subscription',
  'ContentType',
  'WatchStatus',
  '_GenreToUser',
  '_SubscriptionToUser',
  'UserClustered',
]

class BaseModel(SQLModel):
  # If don't use this, table name will be lower case of class name
  @declared_attr
  def __tablename__(cls) -> str:
    return cls.__name__
  
class Config:
  arbitrary_types_allowed = True

class ContentType(enum.Enum):
  MOVIE: int = 0
  TV: int = 1
  
class WatchStatus(enum.Enum):
  WATCHED: int = 0
  WATCHING: int = 1
  WANT_TO_WATCH: int = 2

class User(BaseModel, config=Config, table=True):
  id: str = Field(primary_key=True)
  name: Optional[str] = None
  email: Optional[str] = Field(unique=True)
  emailVerified: Optional[datetime] = None
  image: Optional[str] = None
  avatar: Optional[str] = None
  birth: Optional[datetime] = None
  occupationId: Optional[int] = Field(foreign_key='Occupation.id')
  gender: Optional[str] = None

class Review(BaseModel, config=Config, table=True):
  userId: str = Field(primary_key=True, foreign_key='User.id')
  contentType: ContentType = Field(
    primary_key=True,
    sa_column=Column(
      Enum(ContentType),
    )
  )
  contentId: int = Field(primary_key=True)
  watch: WatchStatus
  rating: int = 0

class Account(BaseModel, config=Config, table=True):
  id: str = Field(primary_key=True)
  userId: str = Field(foreign_key='User.id')
  type: str
  provider: str
  providerAccoundId: str
  refresh_token: Optional[str] = None
  access_token: Optional[str] = None
  expires_at: Optional[int] = None
  token_type: Optional[str] = None
  scope: Optional[str] = None
  id_token: Optional[str] = None
  session_state: Optional[str] = None
  refresh_token_expires_in: Optional[int] = None

class Category(BaseModel, config=Config, table=True):
  id: int = Field(primary_key=True)
  name: str = Field(unique=True)

class CategoriesOnUsers(BaseModel, config=Config, table=True):
  userId: str = Field(primary_key=True, foreign_key='User.id')
  categoryId: int = Field(primary_key=True, foreign_key='Category.id')
  order: int = None

class Occupation(BaseModel, config=Config, table=True):
  id: int = Field(primary_key=True)
  name: str = Field(unique=True)

class Subscription(BaseModel, config=Config, table=True):
  id: int = Field(primary_key=True)
  key: str = Field(unique=True)
  name: str = Field(unique=True)
  providerId: Optional[int] = None
  networkId: Optional[int] = None
  price: int = 0
  sharing: int = 0

class Genre(BaseModel, config=Config, table=True):
  id: int = Field(primary_key=True)
  name: str = Field(unique=True)

class _GenreToUser(BaseModel, config=Config, table=True):
  A: int = Field(primary_key=True, foreign_key='Genre.id')
  B: str = Field(primary_key=True, foreign_key='User.id')

class _SubscriptionToUser(BaseModel, config=Config, table=True):
  A: int = Field(primary_key=True, foreign_key='Subscription.id')
  B: str = Field(primary_key=True, foreign_key='User.id')

class UserClustered(BaseModel, config=Config, table=True):
  id: str = Field(primary_key=True)
  label: int = Field(nullable=False)
