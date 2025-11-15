# Optional: install dependencies first (from your shell, not inside Python):
# pip install torch pandas pyarrow fastparquet scikit-learn nltk huggingface_hub matplotlib regex

from collections import Counter

from matplotlib import pyplot as plt
import nltk
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import regex as re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# --- Step 3: Load dataset from Hugging Face (Deysi/spam-detection-dataset) ---

splits = {
    "train": "data/train-00000-of-00001-daf190ce720b3dbb.parquet",
    "test": "data/test-00000-of-00001-fa9b3e8ade89a333.parquet",
}

df_train = pd.read_parquet("hf://datasets/Deysi/spam-detection-dataset/" + splits["train"])
df_test = pd.read_parquet("hf://datasets/Deysi/spam-detection-dataset/" + splits["test"])

print("Train head:")
print(df_train.head(10))

# --- Text preprocessing: stopwords and cleaning ---

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))


def clean_data(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)


df_train["clean_text"] = df_train["text"].apply(clean_data)
df_test["clean_text"] = df_test["text"].apply(clean_data)

# Map labels: spam -> 1, not_spam -> 0 (float32)
df_train["label"] = df_train["label"].map({"spam": 1, "not_spam": 0}).astype("float32")
df_test["label"] = df_test["label"].map({"spam": 1, "not_spam": 0}).astype("float32")

# --- Step 4: Tokenization and vocabulary ---


def tokenize(text: str):
    return text.split()


train_texts = df_train["clean_text"]
all_tokens = [token for text in train_texts for token in tokenize(text)]
vocab = {word: i + 2 for i, word in enumerate(Counter(all_tokens))}  # start from 2
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1


def encode(text: str, max_len: int = 50):
    tokens = tokenize(text)
    ids = [vocab.get(t, 1) for t in tokens]  # 1 is <UNK>
    if len(ids) < max_len:
        ids += [0] * (max_len - len(ids))  # 0 is <PAD>
    else:
        ids = ids[:max_len]
    return ids


# --- Step 5: Custom Dataset and DataLoaders ---


class SpamDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = [torch.tensor(encode(t), dtype=torch.long) for t in texts]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


train_texts = df_train["clean_text"]
train_labels = df_train["label"].to_numpy(copy=True)
test_texts = df_test["clean_text"]
test_labels = df_test["label"].to_numpy(copy=True)

train_ds = SpamDataset(train_texts, train_labels)
test_ds = SpamDataset(test_texts, test_labels)

train_dl = DataLoader(train_ds, batch_size=32, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=32)

# --- Step 6: Define the model ---


class SpamClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim: int = 64, hidden_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
        )
        self.fc = nn.Linear(in_features=hidden_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        out = self.fc(hidden[-1])
        return self.sigmoid(out).squeeze(1)


vocab_size = len(vocab)
model = SpamClassifier(vocab_size)

# --- Step 7: Training setup ---

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(model)

# --- Training loop ---

epochs = 5
train_losses = []

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X, y in train_dl:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        preds = model(X)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_loss = train_loss / len(train_dl)
    train_losses.append(avg_loss)
    print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f}")

# --- Plot training loss ---

epochs_arr = np.arange(1, len(train_losses) + 1)
plt.figure(figsize=(12, 5))
plt.plot(epochs_arr, train_losses, label="Training Loss", color="blue")
plt.title("Training loss over epochs")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.legend()
plt.show()

# --- Step 8: Evaluation ---

model.eval()
correct, total = 0, 0

with torch.no_grad():
    for X, y in test_dl:
        X, y = X.to(device), y.to(device)
        preds = model(X)
        preds_cls = (preds > 0.5).float()
        correct += (preds_cls == y).sum().item()
        total += y.size(0)

print(f"Final accuracy: {correct / total:.4f}")
