import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():
    # Load dataset
    df = pd.read_csv("dataset_preprocessing.csv")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Set experiment
    mlflow.set_experiment("CI_RFM_Clustering")

    # Train model
    model = KMeans(n_clusters=4, random_state=42)
    model.fit(X_scaled)

    # Evaluation
    labels = model.predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)

    # Manual logging (WAJIB untuk CI)
    mlflow.log_param("n_clusters", 4)
    mlflow.log_metric("silhouette_score", sil_score)
    mlflow.sklearn.log_model(model, "model")

    print("Training selesai via MLflow Project CI")


if __name__ == "__main__":
    main()
