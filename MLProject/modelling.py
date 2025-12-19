import os
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

    # 🔑 Attach ke run MLflow Project
    run_id = os.environ.get("MLFLOW_RUN_ID")

    with mlflow.start_run(run_id=run_id):
        kmeans = KMeans(
            n_clusters=4,
            random_state=42
        )
        kmeans.fit(X_scaled)

        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("inertia", kmeans.inertia_)

        mlflow.sklearn.log_model(
            kmeans,
            artifact_path="kmeans_model"
        )

        print("CI training selesai & artefak tersimpan.")


if __name__ == "__main__":
    main()
