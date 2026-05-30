# =============================================================
# Speak-ALS Project — Phase 3a: SVM Model
# =============================================================
# Support Vector Machine with RBF kernel.
# SVMs work very well on tabular acoustic features.
# We also do hyperparameter tuning with GridSearchCV.
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import joblib

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PLOTS_DIR     = os.path.join(BASE_DIR, "plots")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# Load processed data
# ──────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    print(f"[OK] Data loaded — Train: {X_train.shape}, Test: {X_test.shape}")
    return X_train, X_test, y_train, y_test

# ──────────────────────────────────────────────────────────────
# Hyperparameter tuning with Grid Search
# ──────────────────────────────────────────────────────────────
def tune_svm(X_train, y_train):
    print("[INFO] Running GridSearchCV for SVM hyperparameter tuning...")
    print("       This may take 1-2 minutes...\n")

    param_grid = {
        'C':      [0.1, 1, 10, 100],
        'gamma':  ['scale', 'auto', 0.01, 0.001],
        'kernel': ['rbf', 'linear']
    }

    svm = SVC(probability=True, random_state=42)
    grid_search = GridSearchCV(
        svm, param_grid, cv=5,
        scoring='f1', n_jobs=-1, verbose=1
    )
    grid_search.fit(X_train, y_train)

    print(f"\n[OK] Best parameters : {grid_search.best_params_}")
    print(f"[OK] Best CV F1 score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_, grid_search.best_params_

# ──────────────────────────────────────────────────────────────
# Cross-validation
# ──────────────────────────────────────────────────────────────
def cross_validate(model, X_train, y_train):
    print("\n[INFO] Running 5-fold cross-validation...")
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    print(f"[OK] F1 scores per fold: {np.round(scores, 4)}")
    print(f"[OK] Mean F1: {scores.mean():.4f} ± {scores.std():.4f}")
    return scores

# ──────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, model_name="SVM"):
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  {model_name} — TEST SET RESULTS")
    print(f"{'='*50}")
    print(f"  Accuracy  : {acc:.4f}  ({acc*100:.1f}%)")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}  ← Most important for screening")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"{'='*50}")
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=['Healthy', 'Motor-Impaired']))

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "roc_auc": auc}
    return y_pred, y_prob, metrics

# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────
def plot_confusion_matrix(y_test, y_pred, model_name="SVM"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Healthy', 'Motor-Impaired'],
                yticklabels=['Healthy', 'Motor-Impaired'],
                linewidths=1, linecolor='white',
                annot_kws={"size": 14, "weight": "bold"})
    plt.title(f'{model_name} — Confusion Matrix', fontsize=13, fontweight='bold')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"confusion_matrix_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Confusion matrix saved → {path}")

def plot_roc_curve(y_test, y_prob, model_name="SVM"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#e74c3c', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Random Classifier')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#e74c3c')
    plt.xlabel('False Positive Rate', fontsize=11)
    plt.ylabel('True Positive Rate (Recall)', fontsize=11)
    plt.title(f'{model_name} — ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC curve saved → {path}")

def plot_feature_importance(model, best_params):
    """For linear SVM we can extract weights, for RBF we use permutation importance"""
    if best_params.get('kernel') == 'linear':
        feature_names = [
            'Fo(Hz)', 'Fhi(Hz)', 'Flo(Hz)', 'Jitter%', 'Jitter_abs',
            'RAP', 'PPQ', 'JitterDDP', 'Shimmer', 'Shimmer_dB',
            'APQ3', 'APQ5', 'APQ', 'ShimmerDDA', 'NHR', 'HNR',
            'RPDE', 'DFA', 'spread1', 'spread2', 'D2', 'PPE'
        ]
        coefs = np.abs(model.coef_[0])
        sorted_idx = np.argsort(coefs)[::-1]

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(coefs)), coefs[sorted_idx],
                color='#3498db', edgecolor='white', alpha=0.85)
        plt.xticks(range(len(coefs)), [feature_names[i] for i in sorted_idx],
                   rotation=45, ha='right', fontsize=9)
        plt.title('SVM Feature Importance (Linear Kernel Weights)',
                  fontsize=12, fontweight='bold')
        plt.ylabel('Absolute Weight')
        plt.tight_layout()
        path = os.path.join(PLOTS_DIR, "svm_feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[OK] Feature importance plot saved → {path}")

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SPEAK-ALS — SVM MODEL TRAINING")
    print("="*55 + "\n")

    X_train, X_test, y_train, y_test = load_data()

    # Tune and train
    best_model, best_params = tune_svm(X_train, y_train)

    # Cross-validation
    cross_validate(best_model, X_train, y_train)

    # Evaluate on test set
    y_pred, y_prob, metrics = evaluate(best_model, X_test, y_test, "SVM")

    # Plots
    plot_confusion_matrix(y_test, y_pred, "SVM")
    plot_roc_curve(y_test, y_prob, "SVM")
    plot_feature_importance(best_model, best_params)

    # Save model
    model_path = os.path.join(MODELS_DIR, "svm_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n[OK] Model saved → {model_path}")

    # Save metrics
    import json
    metrics_path = os.path.join(RESULTS_DIR, "svm_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: round(v, 4) for k, v in metrics.items()}, f, indent=2)
    print(f"[OK] Metrics saved → {metrics_path}")

    print("\n[NEXT STEP] Run: python 04_train_cnn.py")
