# src/data/prepare.py
# Purpose: read raw IMDb CSVs, apply minimal cleaning, write processed CSVs.

import html
import os
import re

import pandas as pd
import yaml


def load_params(path: str = "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_text(text: str, cfg: dict) -> str:
    # Ensure string
    text = str(text)
    if cfg.get("lowercase", True):
        text = text.lower()
    if cfg.get("remove_html_tags", True):
        text = re.sub(r"<[^>]+>", " ", text)
    if cfg.get("decode_html_entities", True):
        text = html.unescape(text)
    if cfg.get("normalize_whitespace", True):
        text = re.sub(r"\s+", " ", text).strip()
        # Collapse whitespace introduced before strong punctuation (e.g. from removed HTML tags).
        text = re.sub(r"\s+([?!])", r"\1", text)
    return text


def main():
    params = load_params()
    data_cfg = params["data"]
    prep_cfg = params["preprocessing"]
    schema = data_cfg.get("schema", {"text_col": "text", "label_col": "label"})
    text_col = schema["text_col"]
    label_col = schema["label_col"]

    raw_train = data_cfg["raw"]["train"]
    raw_test = data_cfg["raw"]["test"]
    out_train = data_cfg["processed"]["train"]
    out_test = data_cfg["processed"]["test"]

    os.makedirs(os.path.dirname(out_train), exist_ok=True)

    # Read
    df_train = pd.read_csv(raw_train)
    df_test = pd.read_csv(raw_test)

    # Basic schema validation
    missing_cols = [
        c for c in (text_col, label_col) if c not in df_train.columns or c not in df_test.columns
    ]
    if missing_cols:
        raise ValueError(f"Missing required columns in input CSVs: {missing_cols}")

    # Clean
    df_train[text_col] = df_train[text_col].astype(str).map(lambda s: clean_text(s, prep_cfg))
    df_test[text_col] = df_test[text_col].astype(str).map(lambda s: clean_text(s, prep_cfg))

    # Write
    df_train.to_csv(out_train, index=False)
    df_test.to_csv(out_test, index=False)

    print(f"[prepare] Saved: {out_train}")
    print(f"[prepare] Saved: {out_test}")


if __name__ == "__main__":
    main()
