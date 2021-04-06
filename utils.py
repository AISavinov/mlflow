import sys
import pandas as pd
import mlflow
import numpy as np


def getUsersEvents(user_id, df):
    return np.random.choice(df.loc[df['user_id'] == user_id].event_id.values, 20)


def getFeatures(item_id, col_name, df):
    return df.loc[df[col_name] == item_id]


def pk(actual, predicted):
    assert len(actual) == len(predicted)
    diff = 0
    for a, p in zip(actual, predicted):
        diff += (len(a) - len(list(set(p) - set(a)))) / len(a)
    return diff / len(actual)


def getData():
    train = pd.read_csv('train.csv').drop(columns=['Unnamed: 0'])
    test = pd.read_csv('test.csv').drop(columns=['Unnamed: 0'])
    ml_users = pd.read_csv('ml_users.csv').set_index('user_id').dropna()
    ml_events = pd.read_csv('ml_events.csv').set_index('event_id')
    return train, test, ml_users, ml_events
