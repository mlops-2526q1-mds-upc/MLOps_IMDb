from datasets import load_dataset


def main():
    # Load dataset from Hugging Face
    ds = load_dataset("Deysi/spam-detection-dataset")

    # Save train and test splits to disk as Parquet
    ds["train"].to_parquet("spam_train.parquet")
    ds["test"].to_parquet("spam_test.parquet")

    print("Saved: spam_train.parquet, spam_test.parquet")


if __name__ == "__main__":
    main()
