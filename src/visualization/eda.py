import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import load_dataset

def perform_eda(df):
    plots_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../plots"))
    os.makedirs(plots_dir, exist_ok=True)

    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.shape)
    print(df['Churn'].value_counts())

    plt.figure()
    sns.histplot(data=df, x='Gender', hue='Churn', multiple='dodge', shrink=0.8)
    plt.savefig(os.path.join(plots_dir, "gender_churn.png"))
    plt.show()

    plt.figure()
    sns.histplot(data=df, x='Age', bins=30, kde=True)
    plt.savefig(os.path.join(plots_dir, "age_distribution.png"))
    plt.show()

    plt.figure()
    sns.histplot(data=df, x='Tenure', bins=30, kde=True)
    plt.savefig(os.path.join(plots_dir, "tenure_distribution.png"))
    plt.show()

    plt.figure()
    sns.histplot(data=df, x='Customer Status', hue='Churn', multiple='dodge', shrink=0.8)
    plt.savefig(os.path.join(plots_dir, "customer_status_churn.png"))
    plt.show()

if __name__ == "__main__":
    raw_train_df = load_dataset("data/raw/train.csv")
    raw_test_df = load_dataset("data/raw/test.csv")
    perform_eda(raw_train_df)

