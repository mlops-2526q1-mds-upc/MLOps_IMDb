"""Build vocabulary and token tensors for the spam dataset."""

from collections import Counter
from contextlib import nullcontext
import json
import os
from typing import Dict, Iterable, List

from codecarbon import EmissionsTracker
import pandas as pd
import torch
import yaml


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


def tokenize(text: str) -> List[str]:
    return text.split()


def build_vocab(
    texts: Iterable[str],
    min_freq: int,
    pad_token: str,
    unk_token: str,
) -> Dict[str, int]:
    counter: Counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    vocab = {pad_token: 0, unk_token: 1}
    next_idx = 2
    for token, freq in counter.most_common():
        if freq < min_freq:
            continue
        if token in vocab:
            continue
        vocab[token] = next_idx
        next_idx += 1
    return vocab


def encode_text(text: str, vocab: Dict[str, int], max_len: int, pad_idx: int, unk_idx: int):
    token_ids = [vocab.get(tok, unk_idx) for tok in tokenize(text)]
    if len(token_ids) < max_len:
        token_ids += [pad_idx] * (max_len - len(token_ids))
    else:
        token_ids = token_ids[:max_len]
    return token_ids


def encode_dataset(
    texts: Iterable[str],
    vocab: Dict[str, int],
    max_len: int,
    pad_idx: int,
    unk_idx: int,
) -> torch.Tensor:
    encoded = [encode_text(t, vocab, max_len, pad_idx, unk_idx) for t in texts]
    return torch.tensor(encoded, dtype=torch.long)


def save_tensor_bundle(path: str, inputs: torch.Tensor, labels: torch.Tensor) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"input_ids": inputs, "labels": labels}, path)


def save_vocab(path: str, vocab: Dict[str, int]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def main():
    params = load_params()
    spam_cfg = params["spam"]
    data_cfg = spam_cfg["data"]
    features_cfg = spam_cfg["features"]
    schema = data_cfg.get(
        "schema",
        {"text_col": "text", "label_col": "label", "label_domain": [0, 1]},
    )
    label_col = schema["label_col"]
    clean_col = features_cfg.get("text_column", "clean_text")

    train_path = data_cfg["processed"]["train"]
    test_path = data_cfg["processed"]["test"]
    out_cfg = features_cfg["outputs"]
    train_tensor_path = out_cfg["train_tensor"]
    test_tensor_path = out_cfg["test_tensor"]
    vocab_path = out_cfg["vocab_path"]

    max_len = features_cfg.get("max_len", 50)
    min_freq = features_cfg.get("min_freq", 1)
    pad_token = features_cfg.get("pad_token", "<PAD>")
    unk_token = features_cfg.get("unk_token", "<UNK>")

    with create_tracker(params, "spam_features") as tracker:
        df_train = pd.read_csv(train_path)
        df_test = pd.read_csv(test_path)

        if clean_col not in df_train.columns or clean_col not in df_test.columns:
            raise ValueError(
                f"Expected column '{clean_col}' in processed spam data. "
                "Ensure spam_prepare has been executed."
            )

        vocab = build_vocab(
            df_train[clean_col].astype(str),
            min_freq=min_freq,
            pad_token=pad_token,
            unk_token=unk_token,
        )
        pad_idx = vocab[pad_token]
        unk_idx = vocab[unk_token]

        train_inputs = encode_dataset(
            df_train[clean_col].astype(str),
            vocab,
            max_len=max_len,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
        )
        test_inputs = encode_dataset(
            df_test[clean_col].astype(str),
            vocab,
            max_len=max_len,
            pad_idx=pad_idx,
            unk_idx=unk_idx,
        )

        train_labels = torch.tensor(df_train[label_col].values, dtype=torch.float32)
        test_labels = torch.tensor(df_test[label_col].values, dtype=torch.float32)

        save_tensor_bundle(train_tensor_path, train_inputs, train_labels)
        save_tensor_bundle(test_tensor_path, test_inputs, test_labels)
        save_vocab(vocab_path, vocab)

        print(
            "[spam_features] Train tensor shape:",
            tuple(train_inputs.shape),
            "| Test tensor shape:",
            tuple(test_inputs.shape),
        )
        print(f"[spam_features] Saved tensors -> {train_tensor_path}, {test_tensor_path}")
        print(f"[spam_features] Saved vocab -> {vocab_path}")

    emissions = getattr(tracker, "final_emissions", None)
    if emissions is not None:
        print(f"[emissions] spam_features: {emissions:.6f} kg CO2eq")


if __name__ == "__main__":
    main()
