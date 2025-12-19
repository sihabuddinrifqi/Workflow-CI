import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def main():
    # Load dataset
    df = pd.read_csv("dataset_preprocessing.csv")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Train model
    kmeans = KMeans(
        n_clusters=4,
        random_state=42
    )
    kmeans.fit(X_scaled)

    # ===== MANUAL LOGGING (SKILLED) =====
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
