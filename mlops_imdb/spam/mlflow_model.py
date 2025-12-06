"""MLflow pyfunc wrapper for the spam classifier."""

from __future__ import annotations

import json
from typing import Iterable, List

import mlflow.pyfunc
import pandas as pd
import torch

from mlops_imdb.data.prepare import clean_text
from mlops_imdb.spam.model import SpamClassifier


class SpamPyfuncModel(mlflow.pyfunc.PythonModel):
    """Pyfunc model that accepts raw text and returns spam probabilities."""

    def load_context(self, context):
        with open(context.artifacts["config"], "r", encoding="utf-8") as f:
            self.config = json.load(f)
        with open(context.artifacts["vocab"], "r", encoding="utf-8") as f:
            self.vocab = json.load(f)

        state_dict = torch.load(context.artifacts["state_dict"], map_location="cpu")

        self.pad_token = self.config.get("pad_token", "<PAD>")
        self.unk_token = self.config.get("unk_token", "<UNK>")
        self.max_len = int(self.config.get("max_len", 50))
        self.threshold = float(self.config.get("threshold", 0.5))
        self.pad_idx = int(self.config.get("pad_idx", self.vocab.get(self.pad_token, 0)))
        self.unk_idx = int(self.vocab.get(self.unk_token, 1))
        self.preprocess_cfg = self.config.get("preprocess_cfg", {})

        self.device = torch.device("cpu")
        self.model = SpamClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.config.get("embedding_dim", 64),
            hidden_dim=self.config.get("hidden_dim", 64),
            dropout=self.config.get("dropout", 0.0),
            padding_idx=self.pad_idx,
        )
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _tokenize(self, text: str) -> List[str]:
        return text.split()

    def _encode(self, text: str) -> List[int]:
        token_ids = [self.vocab.get(tok, self.unk_idx) for tok in self._tokenize(text)]
        if len(token_ids) < self.max_len:
            token_ids += [self.pad_idx] * (self.max_len - len(token_ids))
        else:
            token_ids = token_ids[: self.max_len]
        return token_ids

    def _prepare_inputs(self, model_input) -> List[str]:
        if isinstance(model_input, str):
            texts = [model_input]
        elif isinstance(model_input, pd.DataFrame):
            if "text" in model_input.columns:
                texts = model_input["text"].astype(str).tolist()
            else:
                texts = model_input.iloc[:, 0].astype(str).tolist()
        elif isinstance(model_input, pd.Series):
            texts = model_input.astype(str).tolist()
        elif isinstance(model_input, Iterable):
            texts = [str(x) for x in model_input]
        else:
            texts = [str(model_input)]

        cleaned = [clean_text(t, self.preprocess_cfg) for t in texts]
        return cleaned

    def predict(self, context, model_input):
        texts = self._prepare_inputs(model_input)
        encoded = [self._encode(text) for text in texts]
        input_tensor = torch.tensor(encoded, dtype=torch.long, device=self.device)

        with torch.no_grad():
            probs = self.model(input_tensor).cpu().numpy()

        if (
            isinstance(model_input, str)
            or (not hasattr(model_input, "__len__"))
            or len(texts) == 1
        ):
            return float(probs.squeeze())
        return pd.Series(probs.squeeze(), name="prediction")
