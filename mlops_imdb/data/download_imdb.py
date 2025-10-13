from contextlib import nullcontext
import os

from codecarbon import EmissionsTracker
from datasets import load_dataset
import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_tracker(params: dict, project_name: str):
    energy_cfg = params.get("energy", {}).get("codecarbon", {})
    if not energy_cfg.get("enabled"):
        return nullcontext()
    output_path = energy_cfg.get("output", "emissions.csv")
    output_dir, output_file = os.path.split(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."
    return EmissionsTracker(
        project_name=project_name, output_dir=output_dir, output_file=output_file
    )


def main() -> None:
    params = load_params()
    with create_tracker(params, "download_imdb") as tracker:
        ds = load_dataset("imdb")
        os.makedirs("data/raw", exist_ok=True)
        pd.DataFrame(ds["train"]).to_csv("data/raw/imdb_train.csv", index=False)
        pd.DataFrame(ds["test"]).to_csv("data/raw/imdb_test.csv", index=False)
        print("✅ IMDB dataset saved in data/raw/")
    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] download_imdb: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
