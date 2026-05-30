# =============================================================
# Speak-ALS Project — Phase 4: Model Comparison
# =============================================================
# Compares SVM vs CNN, generates final comparison plots,
# and produces a summary results table for the report.
# =============================================================

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
PLOTS_DIR   = os.path.join(BASE_DIR, "plots")

def load_metrics():
    svm_path = os.path.join(RESULTS_DIR, "svm_metrics.json")
    cnn_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")

    with open(svm_path) as f: svm = json.load(f)
    with open(cnn_path) as f: cnn = json.load(f)

    return svm, cnn

def plot_model_comparison(svm_metrics, cnn_metrics):
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels       = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    svm_vals = [svm_metrics[m] for m in metrics_list]
    cnn_vals = [cnn_metrics[m] for m in metrics_list]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, svm_vals, width, label='SVM',
                   color='#3498db', edgecolor='white', alpha=0.9)
    bars2 = ax.bar(x + width/2, cnn_vals, width, label='CNN',
                   color='#e74c3c', edgecolor='white', alpha=0.9)

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_ylim(0, 1.12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title('Speak-ALS: Model Comparison — SVM vs CNN', fontsize=13, fontweight='bold')
    ax.legend(fontsize=11)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, label='0.90 threshold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_facecolor('#f8f9fa')

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Model comparison plot saved → {path}")

def plot_radar_chart(svm_metrics, cnn_metrics):
    """Radar / spider chart for visual comparison"""
    categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    svm_vals = [svm_metrics[m] for m in ['accuracy','precision','recall','f1','roc_auc']]
    cnn_vals = [cnn_metrics[m] for m in ['accuracy','precision','recall','f1','roc_auc']]
    svm_vals += svm_vals[:1]
    cnn_vals += cnn_vals[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, svm_vals, 'o-', lw=2, color='#3498db', label='SVM')
    ax.fill(angles, svm_vals, alpha=0.15, color='#3498db')
    ax.plot(angles, cnn_vals, 's-', lw=2, color='#e74c3c', label='CNN')
    ax.fill(angles, cnn_vals, alpha=0.15, color='#e74c3c')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2','0.4','0.6','0.8','1.0'], fontsize=8)
    ax.set_title('SVM vs CNN — Performance Radar', fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "radar_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Radar chart saved → {path}")

def print_summary_table(svm_metrics, cnn_metrics):
    metrics_list = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    labels       = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']

    print("\n" + "="*55)
    print(f"  {'METRIC':<14}  {'SVM':>10}  {'CNN':>10}  {'WINNER':>8}")
    print("="*55)
    for label, m in zip(labels, metrics_list):
        s, c = svm_metrics[m], cnn_metrics[m]
        winner = "CNN ✓" if c > s else "SVM ✓" if s > c else "TIE"
        print(f"  {label:<14}  {s:>10.4f}  {c:>10.4f}  {winner:>8}")
    print("="*55)

    # Save as CSV for report
    df = pd.DataFrame({
        'Metric': labels,
        'SVM':    [round(svm_metrics[m], 4) for m in metrics_list],
        'CNN':    [round(cnn_metrics[m], 4) for m in metrics_list]
    })
    csv_path = os.path.join(RESULTS_DIR, "model_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n[OK] Results table saved → {csv_path}")

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SPEAK-ALS — MODEL COMPARISON")
    print("="*55 + "\n")

    svm_metrics, cnn_metrics = load_metrics()
    print_summary_table(svm_metrics, cnn_metrics)
    plot_model_comparison(svm_metrics, cnn_metrics)
    plot_radar_chart(svm_metrics, cnn_metrics)

    print("\n[NEXT STEP] Run: python 06_inference_demo.py")
