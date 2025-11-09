# src/data/prepare.py
# Purpose: read raw IMDb CSVs, apply minimal cleaning, write processed CSVs.

from contextlib import nullcontext
import html
import os
from pathlib import Path
import re
import subprocess
from typing import List, Optional, Sequence

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


def ensure_raw_files_exist(raw_files: Sequence[str], remote: Optional[str] = "origin") -> None:
    """Pull raw CSVs with DVC if they are missing locally."""
    missing = [Path(p) for p in raw_files if not Path(p).is_file()]
    if not missing:
        return

    print("[prepare] Raw CSVs missing locally. Pulling from DVC remote...")
    targets: List[str] = []
    for path in missing:
        dvc_pointer = Path(f"{path}.dvc")
        targets.append(str(dvc_pointer if dvc_pointer.is_file() else path))

    cmd = ["dvc", "pull"]
    if remote:
        cmd += ["--remote", remote]
    cmd += targets

    try:
        subprocess.run(cmd, check=True)
    except FileNotFoundError as exc:
        raise RuntimeError(
            "Unable to pull data because the `dvc` CLI is not available in PATH."
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise RuntimeError("`dvc pull` failed. See output above for details.") from exc

    remaining = [str(p) for p in missing if not p.is_file()]
    if remaining:
        raise RuntimeError(f"Failed to retrieve raw data files: {remaining}")


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
        remote = os.environ.get("DVC_REMOTE") or data_cfg.get("remote") or "origin"

        ensure_raw_files_exist([raw_train, raw_test], remote=remote)

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
