import pandas as pd
import pytest

from mlops_imdb.data import prepare


def test_clean_text_applies_preprocessing():
    cfg = {"lowercase": True, "remove_html_tags": True, "normalize_whitespace": True}
    raw = "  Some <b>HTML</b>\nText&nbsp; "
    cleaned = prepare.clean_text(raw, cfg)
    assert cleaned == "some html text&nbsp;"


def test_clean_text_respects_disabled_flags():
    cfg = {"lowercase": False, "remove_html_tags": False, "normalize_whitespace": False}
    raw = " MIXED Case <b>tag</b>\n"
    cleaned = prepare.clean_text(raw, cfg)
    # clean_text always casts to str but should not otherwise modify text
    assert cleaned == raw


def test_main_processes_raw_csvs(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    train_raw = raw_dir / "train.csv"
    test_raw = raw_dir / "test.csv"
    train_processed = processed_dir / "train_clean.csv"
    test_processed = processed_dir / "test_clean.csv"

    pd.DataFrame(
        {
            "text": ["  GREAT <b>Movie</b>!! "],
            "label": [1],
        }
    ).to_csv(train_raw, index=False)
    pd.DataFrame(
        {
            "text": [" bad movie :( "],
            "label": [0],
        }
    ).to_csv(test_raw, index=False)

    params = {
        "data": {
            "raw": {"train": str(train_raw), "test": str(test_raw)},
            "processed": {"train": str(train_processed), "test": str(test_processed)},
            "schema": {"text_col": "text", "label_col": "label"},
        },
        "preprocessing": {
            "lowercase": True,
            "remove_html_tags": True,
            "normalize_whitespace": True,
        },
    }

    monkeypatch.setattr(prepare, "load_params", lambda path="params.yaml": params)
    prepare.main()

    train_out = pd.read_csv(train_processed)
    test_out = pd.read_csv(test_processed)

    assert train_out.loc[0, "text"] == "great movie!!"
    assert test_out.loc[0, "text"] == "bad movie :("


def test_main_raises_for_missing_columns(monkeypatch, tmp_path):
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    raw_dir.mkdir()

    train_raw = raw_dir / "train.csv"
    test_raw = raw_dir / "test.csv"
    train_processed = processed_dir / "train_clean.csv"
    test_processed = processed_dir / "test_clean.csv"

    # Missing label column in train and text column in test to trigger validation
    pd.DataFrame({"text": ["sample only"]}).to_csv(train_raw, index=False)
    pd.DataFrame({"label": [1]}).to_csv(test_raw, index=False)

    params = {
        "data": {
            "raw": {"train": str(train_raw), "test": str(test_raw)},
            "processed": {"train": str(train_processed), "test": str(test_processed)},
            "schema": {"text_col": "text", "label_col": "label"},
        },
        "preprocessing": {},
    }

    monkeypatch.setattr(prepare, "load_params", lambda path="params.yaml": params)

    with pytest.raises(ValueError, match="Missing required columns"):
        prepare.main()


def test_load_params_reads_yaml(tmp_path):
    config_path = tmp_path / "params.yaml"
    config_path.write_text("sample: 123\nnested:\n  value: test\n", encoding="utf-8")

    loaded = prepare.load_params(str(config_path))
    assert loaded == {"sample": 123, "nested": {"value": "test"}}
