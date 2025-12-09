import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.data.load_data import load_dataset

def perform_eda(df):
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())
    print(df.shape)
    print(df['Churn'].value_counts())

    sns.histplot(data=df, x='Gender', hue='Churn', multiple='dodge', shrink=0.8)
    plt.show()

    sns.histplot(data=df, x='Age', bins=30, kde=True)
    plt.show()

    sns.histplot(data=df, x='Tenure', bins=30, kde=True)
    plt.show()

    sns.histplot(data=df, x='Customer Status', hue='Churn', multiple='dodge', shrink=0.8)
    plt.show()

if __name__ == "__main__":
    raw_train_df = load_dataset("data/raw/train.csv")
    raw_test_df = load_dataset("data/raw/test.csv")
    perform_eda(raw_train_df)