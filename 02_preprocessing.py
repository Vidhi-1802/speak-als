# =============================================================
# Speak-ALS Project — Phase 2: Preprocessing Pipeline
# =============================================================
# This script:
#   1. Loads the raw dataset
#   2. Cleans and normalizes features
#   3. Splits into train/test sets
#   4. Saves processed data for model training
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data")
RAW_DIR       = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
PLOTS_DIR     = os.path.join(BASE_DIR, "plots")
MODELS_DIR    = os.path.join(BASE_DIR, "models")

for d in [PROCESSED_DIR, PLOTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── ACOUSTIC FEATURES we use (matching what we'd extract from raw audio) ──
FEATURE_COLS = [
    'MDVP:Fo(Hz)',    # Fundamental frequency - pitch baseline
    'MDVP:Fhi(Hz)',   # Max pitch
    'MDVP:Flo(Hz)',   # Min pitch
    'MDVP:Jitter(%)', # Jitter - freq variation (dysarthria indicator)
    'MDVP:Jitter(Abs)',
    'MDVP:RAP',
    'MDVP:PPQ',
    'Jitter:DDP',
    'MDVP:Shimmer',   # Shimmer - amplitude variation
    'MDVP:Shimmer(dB)',
    'Shimmer:APQ3',
    'Shimmer:APQ5',
    'MDVP:APQ',
    'Shimmer:DDA',
    'NHR',            # Noise-to-Harmonics Ratio
    'HNR',            # Harmonics-to-Noise Ratio
    'RPDE',           # Nonlinear dynamics
    'DFA',
    'spread1',        # Pitch period entropy features
    'spread2',
    'D2',
    'PPE'
]
LABEL_COL = 'status'  # 1 = motor-impaired (ALS-like), 0 = healthy

# ──────────────────────────────────────────────────────────────
# STEP 1: Load data
# ──────────────────────────────────────────────────────────────
def load_data():
    path = os.path.join(RAW_DIR, "parkinsons.csv")
    df = pd.read_csv(path)
    print(f"[OK] Loaded data: {df.shape[0]} samples, {df.shape[1]} columns")
    return df

# ──────────────────────────────────────────────────────────────
# STEP 2: Exploratory analysis plots
# ──────────────────────────────────────────────────────────────
def plot_feature_distributions(df):
    print("[INFO] Plotting feature distributions...")
    key_features = ['MDVP:Jitter(%)', 'MDVP:Shimmer', 'HNR', 'PPE', 'RPDE', 'DFA']

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle('Acoustic Feature Distributions: Impaired vs Healthy', fontsize=14, fontweight='bold')

    for ax, feat in zip(axes.flatten(), key_features):
        healthy  = df[df[LABEL_COL] == 0][feat]
        impaired = df[df[LABEL_COL] == 1][feat]
        ax.hist(healthy,  bins=15, alpha=0.6, color='#2ecc71', label='Healthy',  edgecolor='white')
        ax.hist(impaired, bins=15, alpha=0.6, color='#e74c3c', label='Impaired', edgecolor='white')
        ax.set_title(feat, fontsize=10)
        ax.set_xlabel('Value')
        ax.set_ylabel('Count')
        ax.legend(fontsize=8)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved plot → {path}")

def plot_correlation_heatmap(df):
    print("[INFO] Plotting correlation heatmap...")
    corr = df[FEATURE_COLS + [LABEL_COL]].corr()
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdYlGn', center=0,
                linewidths=0.3, cbar_kws={"shrink": 0.8})
    plt.title('Feature Correlation Matrix', fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved plot → {path}")

def plot_class_balance(df):
    fig, ax = plt.subplots(figsize=(5, 4))
    counts = df[LABEL_COL].value_counts()
    colors = ['#2ecc71', '#e74c3c']
    bars = ax.bar(['Healthy (0)', 'Motor-Impaired (1)'], counts.values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, str(val),
                ha='center', va='bottom', fontweight='bold')
    ax.set_title('Class Balance', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Samples')
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "class_balance.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved plot → {path}")

# ──────────────────────────────────────────────────────────────
# STEP 3: Preprocess — normalize + split
# ──────────────────────────────────────────────────────────────
def preprocess(df):
    print("[INFO] Preprocessing features...")

    X = df[FEATURE_COLS].values
    y = df[LABEL_COL].values

    # Normalize using StandardScaler (zero mean, unit variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Save scaler — needed for inference later
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    joblib.dump(scaler, scaler_path)
    print(f"[OK] Scaler saved → {scaler_path}")

    # Train/test split (80/20, stratified to preserve class balance)
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"[OK] Train set: {X_train.shape[0]} samples")
    print(f"[OK] Test  set: {X_test.shape[0]} samples")

    # Save processed splits
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)

    # Also save feature names for later reference
    pd.Series(FEATURE_COLS).to_csv(os.path.join(PROCESSED_DIR, "feature_names.csv"), index=False)

    print(f"[OK] Processed data saved → {PROCESSED_DIR}")
    return X_train, X_test, y_train, y_test

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SPEAK-ALS — PREPROCESSING PIPELINE")
    print("="*55 + "\n")

    df = load_data()

    print("[INFO] Generating exploratory plots...")
    plot_feature_distributions(df)
    plot_correlation_heatmap(df)
    plot_class_balance(df)

    X_train, X_test, y_train, y_test = preprocess(df)

    print("\n[SUMMARY]")
    print(f"  Features used    : {len(FEATURE_COLS)}")
    print(f"  Training samples : {len(X_train)}")
    print(f"  Testing  samples : {len(X_test)}")
    print(f"  Class distribution (train): {np.bincount(y_train)}")
    print(f"  Class distribution (test) : {np.bincount(y_test)}")

    print("\n[NEXT STEP] Run: python 03_train_svm.py")
