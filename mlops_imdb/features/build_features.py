# src/features/build_features.py
# Purpose: build TF-IDF features for train and test sets based on cleaned text data.

from contextlib import nullcontext
import os

from codecarbon import EmissionsTracker
import joblib
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Load parameters from params.yaml."""
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


def main():
    # Load configuration
    params = load_params()
    with create_tracker(params, "build_features") as tracker:
        features_cfg = params["features"]["tfidf"]
        outputs = params["features"]["outputs"]
        data_cfg = params["data"]

        # Input paths
        train_path = data_cfg["processed"]["train"]
        test_path = data_cfg["processed"]["test"]

        # Output paths
        out_train = outputs["train_features"]
        out_test = outputs["test_features"]
        vectorizer_path = outputs["vectorizer_path"]

        os.makedirs(os.path.dirname(out_train), exist_ok=True)
        os.makedirs(os.path.dirname(vectorizer_path), exist_ok=True)

        # Load datasets
        print(f"[features] Loading cleaned datasets: {train_path}, {test_path}")
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        # TF-IDF vectorizer setup
        print("[features] Building TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(
            max_features=features_cfg.get("max_features", 10000),
            ngram_range=tuple(features_cfg.get("ngram_range", [1, 1])),
            min_df=features_cfg.get("min_df", 1),
            max_df=features_cfg.get("max_df", 1.0),
            dtype=float,
            lowercase=False,  # already cleaned
        )

        # Fit and transform
        X_train = vectorizer.fit_transform(df_train["text"])
        X_test = vectorizer.transform(df_test["text"])

        print(f"[features] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

        # Save sparse matrices
        sp.save_npz(out_train, X_train)
        sp.save_npz(out_test, X_test)

        # Save vectorizer
        joblib.dump(vectorizer, vectorizer_path)

        print(f"[features] Saved train features -> {out_train}")
        print(f"[features] Saved test features  -> {out_test}")
        print(f"[features] Saved TF-IDF vectorizer -> {vectorizer_path}")
        print("[features] Done.")
    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] build_features: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
