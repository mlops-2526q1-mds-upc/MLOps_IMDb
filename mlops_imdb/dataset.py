# mlops_imdb/dataset.py
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from datasets import load_dataset
import pandas as pd
from loguru import logger

from mlops_imdb.config import PROCESSED_DATA_DIR


def _save_splits(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train split: keep separate files to match your current CLI
    train_features = train_df[["text"]].rename(columns={"text": "text"})
    train_labels = train_df[["label"]].rename(columns={"label": "label"})

    train_features.to_csv(out_dir / "features.csv", index=False)
    train_labels.to_csv(out_dir / "labels.csv", index=False)

    # Test split
    test_features = test_df[["text"]].rename(columns={"text": "text"})
    test_labels = test_df[["label"]].rename(columns={"label": "label"})

    test_features.to_csv(out_dir / "test_features.csv", index=False)
    test_labels.to_csv(out_dir / "test_labels.csv", index=False)


def main() -> None:
    """
    Download the IMDB dataset (stanfordnlp/imdb) and materialize it into CSV files:
      - data/processed/features.csv         (columns: text)
      - data/processed/labels.csv           (columns: label)
      - data/processed/test_features.csv    (columns: text)
      - data/processed/test_labels.csv      (columns: label)
    """
    logger.info("Loading 'stanfordnlp/imdb' from Hugging Face Datasets...")
    ds = load_dataset("stanfordnlp/imdb")  # splits: train, test

    logger.info("Converting to pandas...")
    train_df = ds["train"].to_pandas()[["text", "label"]]
    test_df = ds["test"].to_pandas()[["text", "label"]]

    logger.info("Saving CSV files into processed directory...")
    _save_splits(train_df, test_df, PROCESSED_DATA_DIR)

    logger.success("IMDB dataset prepared under data/processed/*.csv")


if __name__ == "__main__":
    main()
