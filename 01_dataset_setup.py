# =============================================================
# Speak-ALS Project — Phase 1: Dataset Setup
# =============================================================
# This script downloads the UCI Parkinson's Voice dataset as a
# stand-in while you request access to the TORGO database.
# The features (MFCCs, jitter, shimmer, HNR) are identical in
# structure to what we'll extract from ALS speech data.
# =============================================================

import os
import urllib.request
import pandas as pd
import numpy as np

# ── Directory setup ─────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR  = os.path.join(DATA_DIR, "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ── Download UCI Parkinson's dataset ─────────────────────────
UCI_URL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
SAVE_PATH = os.path.join(RAW_DIR, "parkinsons.csv")

def download_dataset():
    if os.path.exists(SAVE_PATH):
        print("[INFO] Dataset already downloaded.")
        return
    print("[INFO] Downloading UCI Parkinson's Voice Dataset...")
    try:
        urllib.request.urlretrieve(UCI_URL, SAVE_PATH)
        print(f"[OK]   Saved to: {SAVE_PATH}")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        print("       Please download manually from:")
        print("       https://archive.ics.uci.edu/ml/datasets/parkinsons")

def explore_dataset():
    df = pd.read_csv(SAVE_PATH)

    print("\n" + "="*55)
    print("  DATASET OVERVIEW")
    print("="*55)
    print(f"  Rows       : {df.shape[0]}")
    print(f"  Columns    : {df.shape[1]}")
    print(f"  ALS-like   : {df[df['status']==1].shape[0]}  (motor-impaired)")
    print(f"  Healthy    : {df[df['status']==0].shape[0]}  (controls)")
    print("="*55)

    print("\n[KEY ACOUSTIC FEATURES IN DATASET]")
    features = {
        "MDVP:Fo(Hz)"    : "Average vocal fundamental frequency",
        "MDVP:Jitter(%)" : "Frequency variation (instability)",
        "MDVP:Shimmer"   : "Amplitude variation",
        "HNR"            : "Harmonics-to-Noise Ratio",
        "RPDE"           : "Recurrence Period Density Entropy",
        "DFA"            : "Detrended Fluctuation Analysis",
        "PPE"            : "Pitch Period Entropy",
    }
    for feat, desc in features.items():
        print(f"  {feat:<22} → {desc}")

    print("\n[SAMPLE DATA — first 3 rows]")
    print(df.head(3).to_string())

    # Basic statistics
    print("\n[CLASS BALANCE]")
    counts = df['status'].value_counts()
    for label, count in counts.items():
        tag = "Motor-impaired" if label == 1 else "Healthy"
        bar = "█" * (count // 2)
        print(f"  {tag:<18} ({label}): {count:>3}  {bar}")

    return df

def check_missing_values(df):
    print("\n[MISSING VALUES CHECK]")
    missing = df.isnull().sum().sum()
    if missing == 0:
        print("  ✓ No missing values found. Dataset is clean.")
    else:
        print(f"  ✗ Found {missing} missing values. Will handle in preprocessing.")

if __name__ == "__main__":
    download_dataset()
    df = explore_dataset()
    check_missing_values(df)

    print("\n[NEXT STEP] Run: python 02_preprocessing.py")
