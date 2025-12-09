import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.data.load_data import load_dataset
from src.utils.helper_functions import sync_test_columns


def add_features_interaction(df):
    df = df.copy()
    eps = 1e-6

    df['Usage_Per_Tenure'] = df['Usage Frequency'] / (df['Tenure'] + eps)

    df['Premium_Engagement'] = df['Subscription Type_Premium'] * df['Usage Frequency']

    df['Inactive_Short_Contract'] = df['Customer Status_inactive'] *df['Contract Length_Monthly']

    df['Payment_Delay'] = df['Last Payment Date_day'] - df['Last Due Date_day']

    df['Abs_Payment_Delay'] = df['Payment_Delay'].abs()

    return df

if __name__ == "__main__":
    clean_train_df = load_dataset("data/processed/train_df_clean.csv")
    clean_test_df = load_dataset("data/processed/test_df_clean.csv")

    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)

    clean_train_df = add_features_interaction(clean_train_df)
    clean_test_df = add_features_interaction(clean_test_df)
