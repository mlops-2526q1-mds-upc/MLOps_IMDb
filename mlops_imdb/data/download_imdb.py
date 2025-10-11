import os

from datasets import load_dataset
import pandas as pd

ds = load_dataset("imdb")
os.makedirs("data/raw", exist_ok=True)
pd.DataFrame(ds["train"]).to_csv("data/raw/imdb_train.csv", index=False)
pd.DataFrame(ds["test"]).to_csv("data/raw/imdb_test.csv", index=False)
print("✅ IMDB dataset saved in data/raw/")
