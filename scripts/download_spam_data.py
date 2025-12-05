"""Download and prepare spam dataset from HuggingFace datasets."""

import os
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split


def download_spam_dataset():
    """Download SMS spam dataset from HuggingFace and save as parquet files."""
    
    # Create output directory
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Downloading SMS spam dataset from HuggingFace...")
    
    # Load the SMS spam dataset from HuggingFace
    # This is a popular SMS spam collection dataset
    try:
        dataset = load_dataset("sms_spam", split="train")
    except Exception as e:
        print(f"Failed to load from 'sms_spam', trying alternative...")
        # Alternative: use a different spam dataset
        try:
            dataset = load_dataset("SetFit/sms_spam", split="train")
        except Exception as e2:
            print(f"Failed to load alternative dataset: {e2}")
            # Fallback: create from UCI SMS Spam Collection format
            print("Attempting to download from UCI format...")
            return download_from_uci_format()
    
    # Convert to pandas DataFrame
    df = dataset.to_pandas()
    
    # Check column names and map to expected format
    # The dataset might have different column names
    if "sms" in df.columns or "message" in df.columns or "text" in df.columns:
        text_col = "sms" if "sms" in df.columns else ("message" if "message" in df.columns else "text")
        df = df.rename(columns={text_col: "text"})
    
    if "label" not in df.columns:
        # Check for common label column names
        if "category" in df.columns:
            df = df.rename(columns={"category": "label"})
        elif "type" in df.columns:
            df = df.rename(columns={"type": "label"})
        elif "class" in df.columns:
            df = df.rename(columns={"class": "label"})
        else:
            raise ValueError("Could not find label column in dataset")
    
    # Map labels to expected format (spam/not_spam)
    # Common label formats: 0/1, ham/spam, 0/1 as strings, etc.
    unique_labels = df["label"].unique()
    print(f"Found labels: {unique_labels}")
    
    # Normalize labels to spam/not_spam
    if set(unique_labels).issubset({0, 1, "0", "1"}):
        # Binary numeric labels
        label_map = {0: "not_spam", 1: "spam", "0": "not_spam", "1": "spam"}
    elif set(unique_labels).issubset({"ham", "spam"}):
        # Already in ham/spam format
        label_map = {"ham": "not_spam", "spam": "spam"}
    elif set(unique_labels).issubset({"not_spam", "spam"}):
        # Already in correct format
        label_map = {}
    else:
        # Try to infer: assume 0 or "ham" is not_spam, 1 or "spam" is spam
        label_map = {}
        for label in unique_labels:
            label_str = str(label).lower()
            if label_str in ["0", "ham", "not_spam", "legitimate"]:
                label_map[label] = "not_spam"
            elif label_str in ["1", "spam"]:
                label_map[label] = "spam"
            else:
                print(f"Warning: Unknown label '{label}', mapping to 'not_spam'")
                label_map[label] = "not_spam"
    
    if label_map:
        df["label"] = df["label"].map(label_map)
    
    # Ensure we only have spam and not_spam
    if not set(df["label"].unique()).issubset({"spam", "not_spam"}):
        raise ValueError(f"Unexpected labels after mapping: {df['label'].unique()}")
    
    # Ensure text column exists and is string type
    if "text" not in df.columns:
        raise ValueError("Could not find or create 'text' column")
    df["text"] = df["text"].astype(str)
    
    # Select only required columns
    df = df[["text", "label"]]
    
    # Split into train and test sets (80/20 split)
    df_train, df_test = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"]
    )
    
    # Save as parquet files
    train_path = output_dir / "spam_train.parquet"
    test_path = output_dir / "spam_test.parquet"
    
    df_train.to_parquet(train_path, index=False)
    df_test.to_parquet(test_path, index=False)
    
    print(f"Saved training data: {train_path} ({len(df_train)} samples)")
    print(f"Saved test data: {test_path} ({len(df_test)} samples)")
    print(f"Label distribution - Train: {df_train['label'].value_counts().to_dict()}")
    print(f"Label distribution - Test: {df_test['label'].value_counts().to_dict()}")
    
    return train_path, test_path


def download_from_uci_format():
    """Fallback: download SMS spam data in UCI format."""
    import urllib.request
    
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UCI SMS Spam Collection Dataset URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    
    print("Downloading from UCI repository...")
    zip_path = output_dir / "smsspamcollection.zip"
    
    try:
        urllib.request.urlretrieve(url, zip_path)
        print("Downloaded zip file, extracting...")
        
        import zipfile
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(output_dir)
        
        # Read the SMSSpamCollection file (tab-separated)
        data_file = output_dir / "SMSSpamCollection"
        if not data_file.exists():
            # Try alternative location
            data_file = output_dir / "smsspamcollection" / "SMSSpamCollection"
        
        df = pd.read_csv(
            data_file,
            sep="\t",
            header=None,
            names=["label", "text"]
        )
        
        # Map labels: ham -> not_spam, spam -> spam
        df["label"] = df["label"].map({"ham": "not_spam", "spam": "spam"})
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["label"]
        )
        
        # Save as parquet
        train_path = output_dir / "spam_train.parquet"
        test_path = output_dir / "spam_test.parquet"
        
        df_train.to_parquet(train_path, index=False)
        df_test.to_parquet(test_path, index=False)
        
        print(f"Saved training data: {train_path} ({len(df_train)} samples)")
        print(f"Saved test data: {test_path} ({len(df_test)} samples)")
        
        # Clean up
        zip_path.unlink(missing_ok=True)
        data_file.unlink(missing_ok=True)
        
        return train_path, test_path
        
    except Exception as e:
        print(f"Failed to download from UCI: {e}")
        raise


if __name__ == "__main__":
    download_spam_dataset()

