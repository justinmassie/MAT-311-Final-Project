import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import load_dataset

def sync_test_columns(train_df, test_df):
    for col in train_df.columns:
        if col not in test_df.columns and col not in ['CustomerID', 'Churn']:
            test_df[col] = 0
    feature_cols = [col for col in train_df.columns if col != 'Churn']
    return test_df[feature_cols]


if __name__ == "__main__":
    clean_train_df = load_dataset("data/processed/train_df_clean.csv")
    clean_test_df = load_dataset("data/processed/test_df_clean.csv")

    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)