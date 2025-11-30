import torch

from mlops_imdb.spam.model import SpamClassifier


def test_forward_returns_probabilities():
    model = SpamClassifier(vocab_size=10, embed_dim=8, hidden_dim=4, dropout=0.0)
    batch = torch.randint(0, 10, (3, 5))
    outputs = model(batch)
    assert outputs.shape == (3,)
    assert torch.all((0.0 <= outputs) & (outputs <= 1.0))
