import os
import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

def clean_dataset(train_df, test_df):
    train_df = train_df.copy()
    test_df = test_df.copy()

    # Save target + ID
    train_target = train_df["Churn"] if "Churn" in train_df.columns else None
    train_id = train_df["CustomerID"] if "CustomerID" in train_df.columns else None
    test_id = test_df["CustomerID"] if "CustomerID" in test_df.columns else None

    # Convert numeric-like fields
    for df in [train_df, test_df]:
        df["Support Calls"] = pd.to_numeric(df["Support Calls"], errors="coerce")

    # Date columns
    date_cols = ["Last Due Date", "Last Payment Date"]
    for df in [train_df, test_df]:
        for col in date_cols:
            df[col] = pd.to_datetime(df[col], format="%m-%d", errors="coerce")

    # Numeric and categorical columns
    num_cols = [
        "Age", "Tenure", "Usage Frequency", "Support Calls",
        "Payment Delay", "Total Spend", "Last Interaction"
    ]
    cat_cols = [
        "Gender", "Subscription Type",
        "Contract Length", "Customer Status"
    ]

    # Extract month/day
    for df in [train_df, test_df]:
        for col in date_cols:
            df[f"{col}_month"] = df[col].dt.month
            df[f"{col}_day"] = df[col].dt.day

    # Iterative imputation on both train and test
    imputer = IterativeImputer(random_state=42)
    train_df[num_cols] = imputer.fit_transform(train_df[num_cols])
    test_df[num_cols] = imputer.transform(test_df[num_cols])

    # Fill categorical columns with 'Unknown'
    for df in [train_df, test_df]:
        for col in cat_cols:
            df[col] = df[col].fillna("Unknown")

    # Drop original date columns
    for df in [train_df, test_df]:
        df.drop(columns=date_cols, inplace=True)

    # One-hot encode categorical and month columns
    train_df = pd.get_dummies(train_df, columns=cat_cols + [f"{col}_month" for col in date_cols], drop_first=False)
    test_df = pd.get_dummies(test_df, columns=cat_cols + [f"{col}_month" for col in date_cols], drop_first=False)

    # Add expected dummy columns if missing
    expected_dummy_cols = [
        "Gender_Female", "Gender_Male",
        "Subscription Type_Basic", "Subscription Type_Standard", "Subscription Type_Premium",
        "Contract Length_Annual", "Contract Length_Monthly", "Contract Length_Quarterly",
        "Customer Status_active", "Customer Status_inactive",
        "Last Due Date_month_6", "Last Due Date_month_7",
        "Last Payment Date_month_6", "Last Payment Date_month_7"
    ]
    for df in [train_df, test_df]:
        for col in expected_dummy_cols:
            if col not in df.columns:
                df[col] = 0

    # Final column order
    feature_cols = num_cols + expected_dummy_cols + [f"{col}_day" for col in date_cols]
    train_df = train_df[feature_cols]
    test_df = test_df[feature_cols]

    # Reattach target + ID
    train_df.loc[:, "Churn"] = train_target
    train_df.loc[:, "CustomerID"] = train_id
    test_df.loc[:, "CustomerID"] = test_id

    return train_df, test_df

if __name__ == "__main__":
    raw_train_df = pd.read_csv("data/raw/train.csv")
    raw_test_df = pd.read_csv("data/raw/test.csv")

    clean_train_df, clean_test_df = clean_dataset(raw_train_df, raw_test_df)

    os.makedirs("data/processed", exist_ok=True)

    clean_train_df.to_csv("data/processed/train_df_clean.csv", index=False)
    clean_test_df.to_csv("data/processed/test_df_clean.csv", index=False)

    print("\nCleaned datasets saved to data/processed/")