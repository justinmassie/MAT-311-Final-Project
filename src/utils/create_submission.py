import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import pandas as pd
from src.models.dummy_model_class import DummyModel


def create_submission(model, test_df_scaled, test_ids, output_path='submissions/submission.csv'):
    y_proba = model.predict_proba(test_df_scaled)[:, 1]
    
    submission_df = pd.DataFrame({
        "CustomerID": test_ids,
        "Churn": y_proba
    })
    
    submission_df.to_csv(output_path, index=False)
    print(f"Submission saved to {output_path}")

if __name__ == "__main__":
    pass