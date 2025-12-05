import pandas as pd
import pytest

from mlops_imdb.spam import prepare


def test_main_cleans_and_maps_labels(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    train_raw = raw_dir / "train.parquet"
    test_raw = raw_dir / "test.parquet"
    train_out = processed_dir / "spam_train_clean.csv"
    test_out = processed_dir / "spam_test_clean.csv"

    pd.DataFrame(
        {
            "text": [" Hello!!! <b>WORLD</b> "],
            "label": ["spam"],
        }
    ).to_parquet(train_raw)
    pd.DataFrame(
        {
            "text": ["No spam here."],
            "label": ["not_spam"],
        }
    ).to_parquet(test_raw)

    params = {
        "spam": {
            "data": {
                "raw": {"train": str(train_raw), "test": str(test_raw)},
                "processed": {"train": str(train_out), "test": str(test_out)},
                "schema": {
                    "text_col": "text",
                    "label_col": "label",
                    "label_mapping": {"spam": 1, "not_spam": 0},
                },
            },
            "preprocessing": {
                "lowercase": True,
                "remove_html_tags": True,
                "normalize_whitespace": True,
                "decode_html_entities": True,
            },
        }
    }

    monkeypatch.setattr(prepare, "load_params", lambda path="params.yaml": params)
    monkeypatch.setattr(prepare, "ensure_raw_files_exist", lambda files, remote=None: None)

    prepare.main()

    train_clean = pd.read_csv(train_out)
    test_clean = pd.read_csv(test_out)

    assert "clean_text" in train_clean.columns
    assert train_clean.loc[0, "clean_text"] == "hello!!! world"
    assert pytest.approx(train_clean.loc[0, "label"], rel=1e-6) == 1.0
    assert pytest.approx(test_clean.loc[0, "label"], rel=1e-6) == 0.0


def test_main_raises_for_missing_columns(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()
    processed_dir.mkdir()

    train_raw = raw_dir / "train.parquet"
    test_raw = raw_dir / "test.parquet"
    train_out = processed_dir / "spam_train_clean.csv"
    test_out = processed_dir / "spam_test_clean.csv"

    # Missing label column in train and text column in test
    pd.DataFrame({"text": ["only text"]}).to_parquet(train_raw)
    pd.DataFrame({"label": ["spam"]}).to_parquet(test_raw)

    params = {
        "spam": {
            "data": {
                "raw": {"train": str(train_raw), "test": str(test_raw)},
                "processed": {"train": str(train_out), "test": str(test_out)},
                "schema": {"text_col": "text", "label_col": "label"},
            },
            "preprocessing": {},
        }
    }

    monkeypatch.setattr(prepare, "load_params", lambda path="params.yaml": params)
    monkeypatch.setattr(prepare, "ensure_raw_files_exist", lambda files, remote=None: None)

    with pytest.raises(ValueError, match="Missing columns"):
        prepare.main()


def test_map_labels_errors_on_unknown_values():
    df = pd.DataFrame({"label": ["spam", "maybe"]})
    mapping = {"spam": 1, "not_spam": 0}
    with pytest.raises(ValueError, match="Encountered labels"):
        prepare.map_labels(df, "label", mapping)
