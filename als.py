import implicit
from scipy import sparse

from utils import *
import sys

import mlflow
from sklearn.neighbors import NearestNeighbors

import pandas as pd

if __name__ == "__main__":
    train, test, ml_users, ml_events = getData()
    train.reset_index(inplace=True)
    train['value'] = [1] * train.shape[0]
    train['event_id'] = train['event_id'].astype("category")
    train['user_id'] = train['user_id'].astype("category")

    sparse_event_user_train = sparse.csr_matrix((train['value'].astype(float),
                                                 (train['event_id'].cat.codes,
                                                  train['user_id'].cat.codes)))

    mlflow.create_experiment("test1")
    with mlflow.start_run():
        als_model = implicit.als.AlternatingLeastSquares(factors=20,
                                                         regularization=0.1,
                                                         iterations=2000)

        als_model.fit(sparse_event_user_train)
        mlflow.sklearn.log_model(als_model, 'als_model')
