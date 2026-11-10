from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve
from sklearn.tree import plot_tree


def plot_feature_importances(feature_names: np.ndarray, importance: np.ndarray):
    plt.figure(figsize=(12, 6))
    plt.bar(x=feature_names, height=importance)
    plt.title("Feature importances via coefficients")
    plt.xticks(rotation=90)
    plt.tight_layout()


def plot_roc_curves(final_model, X_val_scaled: np.ndarray, y_val, best_mask: np.ndarray):
    y_val_scores = final_model.predict_proba(X_val_scaled[:, best_mask])
    plt.figure(figsize=(8, 6))
    for class_index, class_label in enumerate(final_model.classes_):
        y_val_binary = (y_val == class_label).astype(int)
        fpr, tpr, _ = roc_curve(y_val_binary, y_val_scores[:, class_index])
        class_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {class_label} (AUC = {class_auc:.3f})")

    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-rest ROC curves (validation data)")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()


def plot_best_tree(best_forest_model, feature_names: np.ndarray, feature_mask: np.ndarray):
    plt.figure(figsize=(20, 10))
    plot_tree(
        best_forest_model.estimators_[0],
        feature_names=feature_names[feature_mask],
        class_names=[str(label) for label in best_forest_model.classes_],
        filled=True,
        max_depth=3,
        fontsize=8,
    )
    plt.title("RandomForest tree")
    plt.tight_layout()


def print_model_results(y_true, y_pred):
    print(confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred))


def print_validation_summary(final_model, X_val_scaled, y_val, best_mask):
    print(classification_report(y_val, final_model.predict(X_val_scaled[:, best_mask])))
