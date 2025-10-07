from dotenv import load_dotenv, find_dotenv
import os, mlflow

def setup_mlflow(experiment_name: str | None = None) -> None:
    load_dotenv(find_dotenv(usecwd=True))  # <— ключевая строка

    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        raise RuntimeError("MLFLOW_TRACKING_URI is not set (check your .env at project root)")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name or os.getenv("MLFLOW_EXPERIMENT", "default"))
    print("[MLflow] tracking_uri:", mlflow.get_tracking_uri())
