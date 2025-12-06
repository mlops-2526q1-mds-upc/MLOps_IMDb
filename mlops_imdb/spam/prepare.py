"""Prepare spam dataset for downstream stages."""

from contextlib import nullcontext
import os
from typing import Dict

from codecarbon import EmissionsTracker
import pandas as pd
import yaml

from mlops_imdb.data.prepare import clean_text, ensure_raw_files_exist


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_tracker(params: dict, project_name: str):
    energy_cfg = params.get("energy", {}).get("codecarbon", {})
    if not energy_cfg.get("enabled"):
        return nullcontext()
    output_path = energy_cfg.get("output", "reports/codecarbon_emissions.csv")
    output_dir, output_file = os.path.split(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = "."
    return EmissionsTracker(
        project_name=project_name, output_dir=output_dir, output_file=output_file
    )


def apply_cleaning(df: pd.DataFrame, text_col: str, cfg: Dict) -> pd.Series:
    series = df[text_col].astype(str).map(lambda s: clean_text(s, cfg))
    return series


def map_labels(df: pd.DataFrame, label_col: str, mapping: Dict) -> pd.Series:
    series = df[label_col].map(mapping)
    if series.isnull().any():
        unknown = df.loc[series.isnull(), label_col].unique()
        raise ValueError(f"Encountered labels not present in mapping: {unknown}")
    return series.astype("float32")


def main():
    params = load_params()
    spam_cfg = params["spam"]
    data_cfg = spam_cfg["data"]
    schema = data_cfg.get(
        "schema",
        {
            "text_col": "text",
            "label_col": "label",
            "label_mapping": {"spam": 1, "not_spam": 0},
        },
    )
    text_col = schema["text_col"]
    label_col = schema["label_col"]

    preprocess_cfg = spam_cfg.get("preprocessing", {})
    label_mapping = schema.get("label_mapping", {})

    raw_train = data_cfg["raw"]["train"]
    raw_test = data_cfg["raw"]["test"]
    out_train = data_cfg["processed"]["train"]
    out_test = data_cfg["processed"]["test"]
    remote = os.environ.get("DVC_REMOTE") or data_cfg.get("remote") or "origin"

    ensure_raw_files_exist([raw_train, raw_test], remote=remote)
    os.makedirs(os.path.dirname(out_train), exist_ok=True)

    with create_tracker(params, "spam_prepare") as tracker:
        df_train = pd.read_parquet(raw_train)
        df_test = pd.read_parquet(raw_test)

        missing_cols = [
            c
            for c in (text_col, label_col)
            if c not in df_train.columns or c not in df_test.columns
        ]
        if missing_cols:
            raise ValueError(f"Missing columns in spam raw data: {missing_cols}")

        df_train["clean_text"] = apply_cleaning(df_train, text_col, preprocess_cfg)
        df_test["clean_text"] = apply_cleaning(df_test, text_col, preprocess_cfg)

        if label_mapping:
            df_train[label_col] = map_labels(df_train, label_col, label_mapping)
            df_test[label_col] = map_labels(df_test, label_col, label_mapping)

        df_train.to_csv(out_train, index=False)
        df_test.to_csv(out_test, index=False)
        print(f"[spam_prepare] Saved -> {out_train}")
        print(f"[spam_prepare] Saved -> {out_test}")

    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] spam_prepare: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
