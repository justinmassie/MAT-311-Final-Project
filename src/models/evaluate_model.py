import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from src.visualization.performance import plot_confusion_matrices, plot_performance_comparison, plot_roc_curve
from src.models.dummy_model_class import DummyModel

def evaluate_model(model, X_val, y_val):
    y_val_pred = model.predict(X_val)
    y_val_proba = model.predict_proba(X_val)[:, 1]

    print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation ROC-AUC:", roc_auc_score(y_val, y_val_proba))
    print(classification_report(y_val, y_val_pred))

    return y_val_pred, y_val_proba

def full_evaluation(model, X_val, y_val, name):
    y_pred, y_proba = evaluate_model(model, X_val, y_val)
    plot_confusion_matrices(y_val, y_pred, name)
    plot_performance_comparison(y_val, y_pred, name)
    plot_roc_curve(y_val, y_proba, name)
    return y_pred, y_proba

if __name__ == '__main__':
    pass