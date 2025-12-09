import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import load_dataset
from src.data.split_data import split_dataset
from sklearn.preprocessing import StandardScaler
from src.utils.helper_functions import sync_test_columns
from src.features.build_features import add_features_interaction

DROP_COLS = [
    'CustomerID',
    'Customer Status_inactive',
    'Customer Status_active'
]

def scale_data(X_train, X_val, X_test):
    cols_to_scale = [col for col in X_train.columns 
                     if X_train[col].nunique() > 2]

    scaler = StandardScaler()

    scaler.fit(X_train[cols_to_scale])

    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale]   = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale]  = scaler.transform(X_test[cols_to_scale])

    return X_train, X_val, X_test

if __name__ == "__main__":
    clean_train_df = load_dataset("data/processed/train_df_clean.csv")
    clean_test_df = load_dataset("data/processed/test_df_clean.csv")

    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)

    # Feature interaction leads to overfitting
    clean_train_df = add_features_interaction(clean_train_df)
    clean_test_df = add_features_interaction(clean_test_df)

    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)

    X_train = X_train.drop(columns=DROP_COLS, errors='ignore')
    X_val = X_val.drop(columns=DROP_COLS, errors='ignore')
    clean_test_df = clean_test_df.drop(columns=DROP_COLS, errors='ignore')

    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)

    X_train_scaled, X_val_scaled, test_df_scaled = scale_data(X_train, X_val, clean_test_df)