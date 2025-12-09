import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, 
                             recall_score, f1_score, roc_curve, roc_auc_score)


def plot_confusion_matrices(y_true, y_pred, model_name):
    """Plot confusion matrices for both models."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix: {model_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()


def plot_performance_comparison(y_true, y_pred, model_name):
    """Plot performance metrics (Accuracy, Precision, Recall, F1) for a single model."""
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    scores = [
        accuracy_score(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred),
        f1_score(y_true, y_pred)
    ]
    
    df = pd.DataFrame({'Metric': metrics, 'Score': scores})
    df.plot(x='Metric', y='Score', kind='bar', legend=False, figsize=(6, 4), color='skyblue')
    plt.ylim(0, 1)
    plt.title(f'Model Performance: {model_name}')
    plt.ylabel('Score')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_score, label):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    auc = roc_auc_score(y_true, y_score)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {label} (AUC={auc:.3f})")
    plt.legend()
    plt.tight_layout()
    plt.show()
    return auc

if __name__ == "__main__":
    pass