import os

from dotenv import load_dotenv
import mlflow
from mlflow.tracking import MlflowClient

load_dotenv()
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

client = MlflowClient()
exp_ids = [e.experiment_id for e in client.search_experiments()]
df = mlflow.search_runs(experiment_ids=exp_ids)

print(
    df[["experiment_id", "run_id", "tags.mlflow.runName", "metrics.accuracy", "start_time"]].head()
)

# Filter for evaluate_model runs and get the most recent one
evaluate_runs = df[df["tags.mlflow.runName"] == "evaluate_model"]
if not evaluate_runs.empty:
    most_recent = evaluate_runs.sort_values("start_time", ascending=False).iloc[0]
    print("Most recent evaluate_model run:")
    print(f"Run ID: {most_recent['run_id']}")
    print(f"Accuracy: {most_recent['metrics.accuracy']}")
    print(f"Date: {most_recent['start_time']}")
else:
    print("No evaluate_model runs found")
