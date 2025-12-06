"""PyTorch model definition for the spam classifier."""

from typing import Optional

import torch.nn as nn


class SpamClassifier(nn.Module):
    """Simple LSTM-based classifier that mirrors spam_training.py."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        dropout: float = 0.0,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.dropout_layer: Optional[nn.Dropout] = (
            nn.Dropout(dropout) if dropout and dropout > 0.0 else None
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        embeddings = self.embedding(x)
        _, (hidden, _) = self.lstm(embeddings)
        features = hidden[-1]
        if self.dropout_layer is not None:
            features = self.dropout_layer(features)
        logits = self.fc(features)
        return self.sigmoid(logits).squeeze(1)
