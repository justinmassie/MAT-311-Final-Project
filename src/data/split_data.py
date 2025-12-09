import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sklearn.model_selection import train_test_split
from src.data.load_data import load_dataset
from src.utils.helper_functions import sync_test_columns
from src.features.build_features import add_features_interaction


def split_dataset(train_df, train_frac=0.7, val_frac=0.3, seed=42):
    if not abs(train_frac + val_frac - 1.0) < 1e-8:
        raise ValueError("Fractions must sum to 1.0")

    y = train_df["Churn"]
    X = train_df.drop(columns=["Churn", "CustomerID"])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=train_frac, random_state=seed, stratify=y
    )

    return (X_train, X_val, y_train, y_val)


if __name__ == "__main__":
    clean_train_df = load_dataset("data/processed/train_df_clean.csv")
    clean_test_df = load_dataset("data/processed/test_df_clean.csv")

    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)

    # Feature interaction leads to overtfitting
    clean_train_df = add_features_interaction(clean_train_df)
    clean_test_df = add_features_interaction(clean_test_df)

    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)
