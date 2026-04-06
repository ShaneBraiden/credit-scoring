"""
Model evaluation and visualization utilities.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
)


def evaluate_model(model, X_test, y_test, model_name: str = "Model") -> dict:
    """
    Comprehensive model evaluation with multiple metrics.
    
    Args:
        model: Trained model with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        model_name: Name for display
        
    Returns:
        Dictionary of metrics
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
    }
    
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1 Score:           {metrics['f1']:.4f}")
    print(f"ROC-AUC:            {metrics['roc_auc']:.4f}")
    print(f"Avg Precision (PR): {metrics['avg_precision']:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))
    
    return metrics


def plot_roc_curve(models_data: list, save_path: str = None):
    """
    Plot ROC curves for multiple models.
    
    Args:
        models_data: List of tuples (model_name, y_test, y_prob)
        save_path: Optional path to save the figure
    """
    plt.figure(figsize=(10, 8))
    
    for model_name, y_test, y_prob in models_data:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})", linewidth=2)
    
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier", linewidth=1)
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.ylabel("True Positive Rate", fontsize=12)
    plt.title("ROC Curve Comparison", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"ROC curve saved to {save_path}")
    
    plt.tight_layout()
    plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name: str, save_path: str = None):
    """
    Plot confusion matrix heatmap.
    
    Args:
        y_test: True labels
        y_pred: Predicted labels
        model_name: Model name for title
        save_path: Optional path to save the figure
    """
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {model_name}", fontsize=14)
    plt.colorbar()
    
    classes = ["No Default", "Default"]
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=11)
    plt.yticks(tick_marks, classes, fontsize=11)
    
    # Add text annotations
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)
    
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Confusion matrix saved to {save_path}")
    
    plt.tight_layout()
    plt.close()


def plot_feature_importance(
    model,
    feature_names: list,
    top_n: int = 15,
    save_path: str = None
):
    """
    Plot feature importance for tree-based models or coefficients for linear models.
    
    Args:
        model: Trained model
        feature_names: List of feature names
        top_n: Number of top features to show
        save_path: Optional path to save the figure
    """
    # Get importance values
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
        importance_type = "Feature Importance"
    elif hasattr(model, "coef_"):
        importance = np.abs(model.coef_[0])
        importance_type = "Coefficient Magnitude"
    else:
        print("Model doesn't have feature_importances_ or coef_ attribute")
        return
    
    # Create DataFrame and sort
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importance
    }).sort_values("importance", ascending=True).tail(top_n)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
    plt.xlabel(importance_type, fontsize=12)
    plt.ylabel("Feature", fontsize=12)
    plt.title(f"Top {top_n} Features by {importance_type}", fontsize=14)
    plt.grid(True, alpha=0.3, axis="x")
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Feature importance plot saved to {save_path}")
    
    plt.tight_layout()
    plt.close()


def compare_models(results: list) -> pd.DataFrame:
    """
    Compare multiple model results and return a summary DataFrame.
    
    Args:
        results: List of metric dictionaries from evaluate_model
        
    Returns:
        DataFrame comparing all models
    """
    df = pd.DataFrame(results)
    df = df.set_index("model_name")
    
    # Round for display
    display_df = df.round(4)
    
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    print(display_df.to_string())
    
    # Highlight best model by ROC-AUC
    best_model = df["roc_auc"].idxmax()
    print(f"\n✓ Best model by ROC-AUC: {best_model} ({df.loc[best_model, 'roc_auc']:.4f})")
    
    return df
