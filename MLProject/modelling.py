import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    mlflow.set_experiment("CI_RFM_Clustering")

    df = pd.read_csv("dataset_preprocessing.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_scaled)

    mlflow.log_param("n_clusters", 4)
    mlflow.log_metric("inertia", model.inertia_)

    mlflow.sklearn.log_model(model, artifact_path="kmeans_model")

    print("Training selesai, model tersimpan sebagai artefak MLflow")


if __name__ == "__main__":
    main()
