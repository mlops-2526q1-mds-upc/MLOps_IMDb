import os

from codecarbon import EmissionsTracker
from datasets import load_dataset
import pandas as pd


def main() -> None:
    with EmissionsTracker(project_name="download_imdb") as tracker:
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
