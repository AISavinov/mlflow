import mlflow
from sklearn.neighbors import NearestNeighbors

mlflow.set_experiment("test7")
with mlflow.start_run():
    mlflow.log_param("n_neighbors", 100)
    mlflow.log_param("n_neighbors_rec", 100)
    mlflow.log_metric("accuracy", 100)

    knn_users = NearestNeighbors(n_neighbors=0 + 1, p=2)
    #artifact_path = mlflow.get_artifact_uri()
    #mlflow.sklearn.save_model(knn_users, artifact_path+"/knn_users")
    mlflow.sklearn.log_model(knn_users, "knn_users")