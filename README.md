# MAT-311 Project

## Project layout

```
.
├── main.py                 # Entry point that runs the entire pipeline
├── requirements.txt        # Python dependencies
├── .gitignore              # Specifies files and folders Git should ignore
├── data/
│   ├── processed/          # Cleaned and preprocessed data saved here
│       └── test_df_clean.csv
│       └── train_df_clean.csv
│   └── raw/                # Raw datasets
│       └── test.csv
│       └── train.csv
├── plots/                  # Generated visualizations from EDA
│   └── .gitkeep            # All plots created by eda.py
├── submissions/            # Model output CSVs
│   └── .gitkeep            # Keeps folder tracked
└── src/
    ├── data/
    │   ├── load_data.py    # Load data from CSV
    │   ├── preprocess.py   # Cleans data
    │   ├── scale_data.py   # Scales data using StandardScaler
    │   └── split_data.py   # Splits the data
    ├── features/
    │   └── build_features.py    # Builds feature interactions
    ├── models/
    │   ├── decision_tree.py            # Trains a decision tree model
    │   ├── dumb_model.py               # Trains a dumb model
    │   ├── dummy_model_class.py        # Creates a class for the dumb model
    │   ├── evaluate_model.py           # Evaluates any model
    │   ├── gradient_boosting.py        # Creates a gradient boosting model
    │   ├── knn_model.py                # Creates a KNN model
    │   ├── logistic_regression.py      # Creates a logistic regression model
    │   ├── random_forest.py            # Creates a random forest model
    ├── utils/
    │   └── helper_functions.py        # Stores helper functions
    │   └── create_submission.py       # Creates the submission CSV files
    └── visualization/
        ├── eda.py                    # Performs EDA
        └── performance.py            # Creates performance evaluation plots
```

`main.py` imports the modules inside `src/` and executes them to reproduce the analysis and results. 

## Running the model

Install the dependencies and run the pipeline.

```bash
conda create -n credit_fraud --file requirements.txt
conda activate credit_fraud
python main.py
```

This will load the dataset, perform data cleaning, split the data, scale the data, and train, evaluate, and create a submission file for a selected model.
The submission file will be stored in submissions/
All EDA done will have files saved to plots/

