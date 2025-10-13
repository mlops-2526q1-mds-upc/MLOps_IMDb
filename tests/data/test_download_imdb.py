from pathlib import Path

import pandas as pd

from mlops_imdb.data import download_imdb


def test_download_imdb_saves_expected_files(tmp_path, monkeypatch):
    fake_dataset = {
        "train": [{"text": "good movie", "label": 1}],
        "test": [{"text": "bad movie", "label": 0}],
    }

    def fake_load_dataset(name):
        assert name == "imdb"
        return fake_dataset

    monkeypatch.setattr(download_imdb, "load_dataset", fake_load_dataset)
    monkeypatch.setattr(
        download_imdb,
        "load_params",
        lambda path="params.yaml": {"energy": {"codecarbon": {"enabled": False}}},
    )
    monkeypatch.chdir(tmp_path)

    download_imdb.main()

    train_path = Path("data/raw/imdb_train.csv")
    test_path = Path("data/raw/imdb_test.csv")

    assert train_path.exists()
    assert test_path.exists()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    pd.testing.assert_frame_equal(
        train_df,
        pd.DataFrame(fake_dataset["train"]),
        check_like=True,
    )
    pd.testing.assert_frame_equal(
        test_df,
        pd.DataFrame(fake_dataset["test"]),
        check_like=True,
    )
