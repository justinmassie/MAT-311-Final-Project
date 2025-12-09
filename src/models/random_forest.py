import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from src.data.split_data import split_dataset
from src.data.scale_data import scale_data
from src.utils.helper_functions import sync_test_columns
from src.models.evaluate_model import full_evaluation

DROP_COLS = [
    'CustomerID',
    'Customer Status_inactive',
    'Customer Status_active'
]

def train_random_forest_classifier(X_train, y_train, n_estimators=200, max_depth=None, random_state=42):
    rf_model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight='balanced'
    )

    rf_model.fit(X_train, y_train)
    return rf_model

if __name__ == '__main__':
    clean_train_df = pd.read_csv('data/processed/train_df_clean.csv')
    clean_test_df = pd.read_csv('data/processed/test_df_clean.csv')

    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)

    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)

    test_ids = clean_test_df['CustomerID'].copy()

    X_train = X_train.drop(columns=DROP_COLS, errors='ignore')
    X_val = X_val.drop(columns=DROP_COLS, errors='ignore')
    clean_test_df = clean_test_df.drop(columns=DROP_COLS, errors='ignore')

    X_train_scaled, X_val_scaled, test_df_scaled = scale_data(X_train, X_val, clean_test_df)

    model_name = "Random Forest"

    model = train_random_forest_classifier(X_train_scaled, y_train)

    full_evaluation(model, X_val_scaled, y_val, model_name)