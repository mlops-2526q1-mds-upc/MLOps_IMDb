# src/data/prepare.py
# Purpose: read raw IMDb CSVs, apply minimal cleaning, write processed CSVs.

from contextlib import nullcontext
import html
import os
import re

from codecarbon import EmissionsTracker
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
    with create_tracker(params, "prepare_data") as tracker:
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
            c
            for c in (text_col, label_col)
            if c not in df_train.columns or c not in df_test.columns
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
    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] prepare_data: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
