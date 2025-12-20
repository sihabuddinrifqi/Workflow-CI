import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    # JANGAN gunakan mlflow.set_experiment di sini karena akan menyebabkan 
    # ketidaksinkronan ID saat dijalankan via 'mlflow run'.
    
    df = pd.read_csv("dataset_preprocessing.csv")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)

    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X_scaled)

    # Menggunakan blok start_run() tanpa parameter agar otomatis menyambung
    # ke Run ID yang sudah dibuat oleh perintah 'mlflow run'
    with mlflow.start_run():
        mlflow.log_param("n_clusters", 4)
        mlflow.log_param("random_state", 42)
        mlflow.log_metric("inertia", kmeans.inertia_)

        mlflow.sklearn.log_model(
            kmeans,
            artifact_path="kmeans_model"
        )
        print("Training selesai & artefak tersimpan.")

if __name__ == "__main__":
    main()
