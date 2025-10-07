# mlops_imdb/debug_mlflow.py
from __future__ import annotations
import os
from urllib.parse import urlparse
from dotenv import load_dotenv, find_dotenv
import mlflow
from mlflow.tracking import MlflowClient

def infer_repo_from_uri(uri: str) -> tuple[str, str] | None:
    # expects https://dagshub.com/<USER>/<REPO>.mlflow
    p = urlparse(uri)
    parts = [x for x in p.path.split("/") if x]
    if len(parts) >= 2 and parts[-1].endswith(".mlflow"):
        user = parts[-2]
        repo = parts[-1].removesuffix(".mlflow")
        return user, repo
    return None

def main():
    load_dotenv(find_dotenv(usecwd=True))
    uri = os.getenv("MLFLOW_TRACKING_URI")
    user = os.getenv("MLFLOW_TRACKING_USERNAME")
    token_set = bool(os.getenv("MLFLOW_TRACKING_PASSWORD"))
    exp_name = os.getenv("MLFLOW_EXPERIMENT", "default")
    dagshub_repo = os.getenv("DAGSHUB_REPO")  # optional "<USER>/<REPO>"

    print("ENV tracking URI:", uri)
    print("ENV username set:", bool(user))
    print("ENV token set:", token_set)

    if not uri:
        raise SystemExit("No MLFLOW_TRACKING_URI in env. Check .env location/content.")

    mlflow.set_tracking_uri(uri)
    print("Effective mlflow.get_tracking_uri():", mlflow.get_tracking_uri())

    # Connect client and ensure experiment exists
    client = MlflowClient(tracking_uri=uri)
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
        exp = client.get_experiment(exp_id)
        print(f"Created experiment '{exp_name}' with id {exp.experiment_id}")
    else:
        print(f"Found experiment '{exp.name}' with id {exp.experiment_id}")

    # Start a tiny run
    with mlflow.start_run(run_name="debug_smoke", experiment_id=exp.experiment_id) as run:
        mlflow.log_param("ok", True)
        mlflow.log_metric("ping", 1.0)
        run_id = run.info.run_id
        print("Run ID:", run_id)

    # Build a DagsHub UI URL
    user_repo = dagshub_repo or (lambda x: f"{x[0]}/{x[1]}" if x else None)(infer_repo_from_uri(uri))
    if "dagshub.com" in uri and user_repo:
        link = f"https://dagshub.com/{user_repo}.mlflow/#/experiments/{exp.experiment_id}/runs/{run_id}"
        print("Open in DagsHub:", link)
    else:
        print("Not a DagsHub URI or repo unknown; skip link.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", repr(e))
        print("Hints:")
        print("- Make sure your Personal Access Token is correct and has repo access.")
        print("- Check that the URI is exactly https://dagshub.com/<USER>/<REPO>.mlflow")
        print("- Private repo? Username must be your DagsHub login, password = access token.")
