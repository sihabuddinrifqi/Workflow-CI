import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Jangan set_experiment di sini jika menggunakan MLProject, 
    # biarkan CLI yang menanganinya.
    
    # Load data
    df = pd.read_csv("dataset_preprocessing.csv")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    # Train model
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)

    # Ambil ACTIVE RUN jika ada (dari mlflow run), jika tidak ada buat baru
    active_run = mlflow.active_run()
    
    with (mlflow.start_run(run_id=active_run.info.run_id) if active_run else mlflow.start_run()):
        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("inertia", kmeans.inertia_)

        mlflow.sklearn.log_model(
            kmeans,
            artifact_path="kmeans_model"
        )
        print("Training selesai.")

if __name__ == "__main__":
    main()
