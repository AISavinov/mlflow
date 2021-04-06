from utils import *
import sys

import mlflow
from sklearn.neighbors import NearestNeighbors


if __name__ == "__main__":
    n_neighbors = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    n_neighbors_rec = int(sys.argv[2]) if len(sys.argv) > 2 else 20

    train, test, ml_users, ml_events = getData()

    mlflow.set_experiment("test1")
    with mlflow.start_run():
        knn_users = NearestNeighbors(n_neighbors=n_neighbors + 1, p=2)
        knn_users.fit(ml_users)

        # to drop target user from neighbors
        neighbors = knn_users.kneighbors(ml_users.values, return_distance=False)[..., 1:]

        recs = []
        actual = []
        print('here')
        for user_id, neighbor_indexs in zip(ml_users.index, neighbors):
            t = test.loc[test['user_id'] == user_id].event_id.values
            if len(t) == 100:
                actual.append(t)
                recs.append(np.concatenate([getUsersEvents(ml_users.index[ind], train) for ind in neighbor_indexs]))

        accuracy = pk(actual, recs)
        print("Knn (n_neighbors=%f, n_neighbors_rec=%f):" % (n_neighbors, n_neighbors_rec))
        print("  Accuracy: %s" % accuracy)

        mlflow.log_param("n_neighbors", n_neighbors)
        mlflow.log_param("n_neighbors_rec", n_neighbors_rec)
        mlflow.log_metric("accuracy", accuracy)

        mlflow.sklearn.log_model(knn_users, "knn_users")
