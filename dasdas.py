import pandas as pd

df: pd.DataFrame = pd.read_pickle('./preprocessed_data/graphFeature.pkl')

rename = {
    'Occupation_0.0': 'Occupation_0',
    'Occupation_1.0': 'Occupation_1',
    'Occupation_2.0': 'Occupation_2',
    'Occupation_3.0': 'Occupation_3',
    'Occupation_4.0': 'Occupation_4',
    'Occupation_5.0': 'Occupation_5',
    'Occupation_6.0': 'Occupation_6',
    'Occupation_7.0': 'Occupation_7',
    'Occupation_8.0': 'Occupation_8',
    'Occupation_9.0': 'Occupation_9',
    'Occupation_10.0': 'Occupation_10',
    'Age_1': '0',
    'Age_2': '1',
    'Age_3': '2',
    'Age_4': '3',
    'Age_5': '4',
    'Age_6': '5',
}

df: pd.DataFrame = df.rename(columns=rename)

print(df)

df.to_pickle('./preprocessed_data/graphFeature.pkl')