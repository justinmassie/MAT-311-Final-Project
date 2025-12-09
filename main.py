from src.data.load_data import load_dataset
from src.visualization.eda import perform_eda
from src.data.preprocess import clean_dataset
from src.data.split_data import split_dataset
from src.utils.helper_functions import sync_test_columns
from src.models.random_forest import train_random_forest_classifier
from src.features.build_features import add_features_interaction
from src.data.scale_data import scale_data
from src.models.logistic_regression import train_logistic_regression_model
from src.models.knn_model import train_knn_model
from src.models.decision_tree import train_decision_tree_model
from src.models.gradient_boosting import train_gradient_boosting_model
from src.models.evaluate_model import full_evaluation
from src.utils.create_submission import create_submission
from src.models.dumb_model import train_dumb_model

DROP_COLS = [
    'CustomerID',
    'Customer Status_inactive',
    'Customer Status_active'
]

def main():
    print("---Loading raw data...")
    train_df = load_dataset("data/raw/train.csv")
    test_df = load_dataset("data/raw/test.csv")

    print("---Doing EDA...")
    perform_eda(train_df)

    print("---Cleaning raw data...")
    clean_train_df, clean_test_df = clean_dataset(train_df, test_df)

    print(f"Cleaned train dataset shape: {clean_train_df.shape}")
    print(f"Cleaned test dataset shape: {clean_test_df.shape}")

    # Ensure all test columns exist except 'CustomerID' and 'Churn'
    print("---Syncing test columns with train columns...")
    clean_test_df = sync_test_columns(clean_train_df, clean_test_df)

    # Feature interaction led to overfitting
    """
    print("---Adding interaction features...")
    clean_train_df = add_features_interaction(clean_train_df)
    clean_test_df = add_features_interaction(clean_test_df)
    """

    print("---Splitting data...")
    X_train, X_val, y_train, y_val = split_dataset(clean_train_df)

    test_ids = clean_test_df['CustomerID'].copy()

    # Drop CustomerID for training/validation
    X_train = X_train.drop(columns=DROP_COLS, errors='ignore')
    X_val = X_val.drop(columns=DROP_COLS, errors='ignore')
    clean_test_df = clean_test_df.drop(columns=DROP_COLS, errors='ignore')

    print("---Scaling data...")
    X_train_scaled, X_val_scaled, test_df_scaled = scale_data(X_train, X_val, clean_test_df)

    # Set this value to select model to run
    MODEL_TO_RUN = 'dumb'

    MODELS = {
        'rf': (train_random_forest_classifier, 'Random Forest', 'submissions/rf_submission.csv'),
        'gb': (train_gradient_boosting_model, 'Gradient Boosting', 'submissions/gradient_boosting_submission.csv'),
        'knn': (train_knn_model, 'KNN', 'submissions/knn_submission.csv'),
        'logreg': (train_logistic_regression_model, 'Logistic Regression', 'submissions/logistic_regression_submission.csv'),
        'dt': (train_decision_tree_model, 'Decision Tree', 'submissions/decision_tree_submission.csv'),
        'dumb': (train_dumb_model, 'Dumb Model', 'submissions/dumb_model_submission.csv'),
    }

    train_fn, model_name, submission_file = MODELS[MODEL_TO_RUN]

    print(f"---Training {model_name} model...")
    model = train_fn(X_train_scaled, y_train)

    print(f"---Evaluating {model_name} model...")
    full_evaluation(model, X_val_scaled, y_val, model_name)

    print(f"---Creating submission file for {model_name}...")
    create_submission(model, test_df_scaled, test_ids, submission_file)

    print("Done.")


if __name__ == "__main__":
    main()
