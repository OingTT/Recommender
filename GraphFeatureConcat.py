import os

import pandas as pd

from apps.arg_parser import get_args
from apps.utils.utils import save_pickle, load_pickle
from apps.GHRS.DataBaseLoader import DataBaseLoader
from apps.GHRS.GraphFeature.GraphFeature_GraphTool import GraphFeature_GraphTool as GraphFeature

ML_1M = './ml-1m'

ml_users = pd.read_csv(
    os.path.join(ML_1M, 'users.dat'),
    sep='\::',
    engine='python',
    names=['UID', 'Gender', 'Age', 'Occupation', 'Zip'],
    dtype={
    'UID': 'str',
    'Gender': 'str',
    'Age': 'uint8',
    'Occupation': 'uint8',
    'Zip': 'string'
    }
)
ml_ratings = pd.read_csv(
    os.path.join(ML_1M, 'ratings.dat'),
    sep='\::',
    engine='python',
    names=['UID', 'MID', 'Rating', 'Timestamp'],
    dtype={
    'UID': 'str',
    'MID': 'uint16',
    'Rating': 'uint8',
    'Timestamp': 'uint64'
    }
)

dataBaseLoader = DataBaseLoader()

db_users = dataBaseLoader.getAllUsers()
db_ratings = dataBaseLoader.getAllReviews()

all_users = pd.concat([ml_users, db_users], axis=0)
all_ratings = pd.concat([ml_ratings, db_ratings], axis=0)


rate = int(len(all_users) / 10)
rates = 0
for i in range(10):
    rates += rate
    if i >= 9 and rates != len(all_users):
        rate += (len(all_users) - rates)
    
    graphFeatures_prev = load_pickle('./preprocessed_data/graphFeature.pkl')

    user_df = all_users.sample(rate)
    rating_list = list()
    for uid in user_df['UID']:
        rating = all_ratings[all_ratings['UID'] == uid]

        rating_list.append(rating)

    rating_df = pd.concat(rating_list, axis=0)

    ghrs = GraphFeature(users_df=user_df, ratings_df=rating_df)
    graphFeatures = ghrs()

    if isinstance(graphFeatures_prev, pd.DataFrame):
        graphFeatures = pd.concat([graphFeatures_prev, graphFeatures], axis=0)
    print(graphFeatures)
    save_pickle(graphFeatures, './preprocessed_data/graphFeature.pkl')