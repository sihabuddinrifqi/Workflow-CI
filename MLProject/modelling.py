import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


def main():
    # Set tracking ke local folder
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("CI_RFM_Clustering")

    # PAKSA start run baru (CI-safe)
    with mlflow.start_run(run_name="ci_training_run"):
        # Load dataset
        df = pd.read_csv("dataset_preprocessing.csv")

        # Scaling
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df)

        # Train model
        model = KMeans(n_clusters=4, random_state=42)
        model.fit(X_scaled)

        # Evaluation
        labels = model.predict(X_scaled)
        sil_score = silhouette_score(X_scaled, labels)

        # Manual logging (Skilled)
        mlflow.log_param("n_clusters", 4)
        mlflow.log_metric("silhouette_score", sil_score)
        mlflow.sklearn.log_model(model, "model")

        print("Training CI selesai & artefak tersimpan")


if __name__ == "__main__":
    main()
