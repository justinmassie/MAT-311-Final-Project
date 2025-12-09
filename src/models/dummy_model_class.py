import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import numpy as np

class DummyModel:
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self
    
    def predict(self, X):
        if 'Gender_Male' not in X.columns:
            raise ValueError('X must contain Gender_Male column.')
        

        return np.where(X['Gender_Male'] == 1, 1, 0)
    
    def predict_proba(self, X):
        preds = self.predict(X)
        prob_1 = preds.astype(float)
        prob_0 = 1 - prob_1
        return np.vstack([prob_0, prob_1]).T