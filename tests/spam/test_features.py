import json

import pandas as pd
import torch

from mlops_imdb.spam import features


def test_build_vocab_respects_min_freq():
    vocab = features.build_vocab(
        ["alpha beta beta", "gamma gamma gamma"],
        min_freq=2,
        pad_token="<PAD>",
        unk_token="<UNK>",
    )
    assert vocab["<PAD>"] == 0
    assert vocab["<UNK>"] == 1
    assert "alpha" not in vocab  # appears once only
    assert "beta" in vocab
    assert "gamma" in vocab


def test_main_generates_tensors_and_vocab(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    processed_dir.mkdir()

    train_csv = processed_dir / "spam_train_clean.csv"
    test_csv = processed_dir / "spam_test_clean.csv"

    train_df = pd.DataFrame({"clean_text": ["hello spam", "not spam"], "label": [1.0, 0.0]})
    test_df = pd.DataFrame({"clean_text": ["hello"], "label": [1.0]})

    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    out_train = tmp_path / "spam_train_features.pt"
    out_test = tmp_path / "spam_test_features.pt"
    vocab_path = tmp_path / "spam_vocab.json"

    params = {
        "spam": {
            "data": {
                "processed": {"train": str(train_csv), "test": str(test_csv)},
                "schema": {"label_col": "label"},
            },
            "features": {
                "max_len": 4,
                "min_freq": 1,
                "pad_token": "<PAD>",
                "unk_token": "<UNK>",
                "text_column": "clean_text",
                "outputs": {
                    "train_tensor": str(out_train),
                    "test_tensor": str(out_test),
                    "vocab_path": str(vocab_path),
                },
            },
        }
    }

    monkeypatch.setattr(features, "load_params", lambda path="params.yaml": params)

    features.main()

    assert out_train.exists()
    assert out_test.exists()
    assert vocab_path.exists()

    train_bundle = torch.load(out_train)
    test_bundle = torch.load(out_test)

    assert train_bundle["input_ids"].shape == (2, 4)
    assert torch.equal(train_bundle["labels"], torch.tensor([1.0, 0.0]))
    assert test_bundle["input_ids"].shape == (1, 4)

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)
    assert "hello" in vocab_data
    assert vocab_data["<PAD>"] == 0
