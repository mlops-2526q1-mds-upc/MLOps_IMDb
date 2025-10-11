import runpy
import sys
import types
from pathlib import Path

import pandas as pd


def test_download_imdb_saves_expected_files(tmp_path, monkeypatch):
    fake_dataset = {
        "train": [{"text": "good movie", "label": 1}],
        "test": [{"text": "bad movie", "label": 0}],
    }

    fake_datasets = types.ModuleType("datasets")

    def fake_load_dataset(name):
        assert name == "imdb"
        return fake_dataset

    fake_datasets.load_dataset = fake_load_dataset

    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)
    monkeypatch.chdir(tmp_path)

    module_name = "mlops_imdb.data.download_imdb"
    original_module = sys.modules.get(module_name)
    try:
        sys.modules.pop(module_name, None)
        runpy.run_module(module_name)
    finally:
        if original_module is not None:
            sys.modules[module_name] = original_module
        else:
            sys.modules.pop(module_name, None)

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
