import pandas as pd

def load_dataset(file_path):
    return pd.read_csv(file_path)


if __name__ == "__main__":
    train_df = load_dataset("data/raw/train.csv")
    test_df = load_dataset("data/raw/test.csv")
    print(test_df.head())
    print(train_df.head())