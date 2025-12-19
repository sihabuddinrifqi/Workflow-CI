import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():
    # Load dataset hasil preprocessing
    df = pd.read_csv("dataset_preprocessing.csv")

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Set experiment MLflow
    mlflow.set_experiment("RFM_Clustering_Basic")

    # Autolog (parameter & model)
    mlflow.sklearn.autolog(log_models=True)

    with mlflow.start_run():
        model = KMeans(
            n_clusters=4,
            random_state=42
        )

        # Train model
        model.fit(X_scaled)

        # ================= METRIC WAJIB =================
        labels = model.predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        # Log metric manual (WAJIB untuk clustering)
        mlflow.log_metric("silhouette_score", sil_score)

        print("Model clustering berhasil dilatih.")
        print(f"Silhouette Score: {sil_score:.4f}")


if __name__ == "__main__":
    main()
