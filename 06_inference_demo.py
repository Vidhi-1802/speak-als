# =============================================================
# Speak-ALS Project — Phase 5: Inference Demo (CLI Prototype)
# =============================================================
# This is the final demo script that:
#   - Takes a new sample (simulated or real feature vector)
#   - Runs it through both SVM and CNN
#   - Outputs a risk indication with explanation
#
# This represents the "screening tool" aspect of the project.
# =============================================================

import os
import numpy as np
import joblib
import json

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR   = os.path.join(BASE_DIR, "data", "processed")

FEATURE_NAMES = [
    'Fundamental Frequency (Hz)', 'Max Pitch (Hz)', 'Min Pitch (Hz)',
    'Jitter (%)', 'Jitter (Abs)', 'RAP', 'PPQ', 'Jitter DDP',
    'Shimmer', 'Shimmer (dB)', 'APQ3', 'APQ5', 'APQ', 'Shimmer DDA',
    'NHR', 'HNR (dB)', 'RPDE', 'DFA', 'Spread1', 'Spread2', 'D2', 'PPE'
]

# ── Risk interpretation ────────────────────────────────────────
def interpret_risk(svm_prob, cnn_prob):
    ensemble_prob = (svm_prob + cnn_prob) / 2.0

    if ensemble_prob >= 0.75:
        level = "HIGH RISK"
        color = "🔴"
        advice = "Strong acoustic indicators of speech motor impairment detected. Prompt neurological evaluation recommended."
    elif ensemble_prob >= 0.50:
        level = "MODERATE RISK"
        color = "🟡"
        advice = "Some acoustic abnormalities detected. Follow-up with a speech-language pathologist is advised."
    elif ensemble_prob >= 0.30:
        level = "LOW-MODERATE RISK"
        color = "🟠"
        advice = "Minor irregularities detected. Monitor over time and consult if symptoms worsen."
    else:
        level = "LOW RISK"
        color = "🟢"
        advice = "No significant acoustic markers of motor speech impairment detected."

    return level, color, advice, ensemble_prob

# ── Load models ────────────────────────────────────────────────
def load_models():
    scaler    = joblib.load(os.path.join(MODELS_DIR, "scaler.pkl"))
    svm_model = joblib.load(os.path.join(MODELS_DIR, "svm_model.pkl"))

    # Load CNN
    import tensorflow as tf
    savedmodel_path = os.path.join(MODELS_DIR, "cnn_savedmodel")
    h5_path = os.path.join(MODELS_DIR, "cnn_final.h5")
    if os.path.exists(savedmodel_path):
        cnn_model = tf.saved_model.load(savedmodel_path)
        cnn_model.predict = lambda x, **kw: cnn_model(x.astype("float32"), training=False).numpy()
    else:
        cnn_model = tf.keras.models.load_model(h5_path)

    return scaler, svm_model, cnn_model

# ── Predict ────────────────────────────────────────────────────
def predict(features_raw, scaler, svm_model, cnn_model):
    # Normalize
    features_scaled = scaler.transform([features_raw])

    # SVM prediction
    svm_prob = svm_model.predict_proba(features_scaled)[0][1]
    svm_pred = int(svm_prob >= 0.5)

    # CNN prediction
    cnn_input = features_scaled.reshape(1, features_scaled.shape[1], 1)
    cnn_prob  = float(cnn_model.predict(cnn_input, verbose=0)[0][0])
    cnn_pred  = int(cnn_prob >= 0.5)

    return svm_prob, svm_pred, cnn_prob, cnn_pred

# ── Demo with real test samples ────────────────────────────────
def run_demo(scaler, svm_model, cnn_model):
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Denormalize to get "original" feature values for display
    X_original = scaler.inverse_transform(X_test)

    print("\n" + "═"*60)
    print("   SPEAK-ALS SCREENING PROTOTYPE — DEMO")
    print("═"*60)
    print("   ⚠  DISCLAIMER: This is a RESEARCH PROTOTYPE only.")
    print("      It is NOT a medical diagnostic tool.")
    print("═"*60)

    # Run 3 demo samples: 1 healthy, 1 impaired, 1 borderline
    demo_indices = []

    # Find a healthy sample
    healthy_idx = np.where(y_test == 0)[0]
    if len(healthy_idx) > 0:
        demo_indices.append(("HEALTHY CONTROL", healthy_idx[0]))

    # Find an impaired sample
    impaired_idx = np.where(y_test == 1)[0]
    if len(impaired_idx) > 0:
        demo_indices.append(("MOTOR-IMPAIRED", impaired_idx[0]))

    for label, idx in demo_indices:
        features_raw = X_original[idx]
        actual_label = "Motor-Impaired" if y_test[idx] == 1 else "Healthy"

        svm_prob, svm_pred, cnn_prob, cnn_pred = predict(
            features_raw, scaler, svm_model, cnn_model
        )

        risk_level, risk_color, advice, ensemble_prob = interpret_risk(svm_prob, cnn_prob)

        print(f"\n{'─'*60}")
        print(f"  SAMPLE TYPE   : {label}")
        print(f"  ACTUAL CLASS  : {actual_label}")
        print(f"{'─'*60}")

        print(f"\n  KEY ACOUSTIC FEATURES:")
        key_features = [0, 3, 8, 15, 21]  # Fo, Jitter%, Shimmer, HNR, PPE
        for i in key_features:
            print(f"    {FEATURE_NAMES[i]:<32}: {features_raw[i]:.4f}")

        print(f"\n  MODEL OUTPUTS:")
        print(f"    SVM  Risk Probability : {svm_prob:.4f}  → {'⚠ IMPAIRED' if svm_pred else '✓ HEALTHY'}")
        print(f"    CNN  Risk Probability : {cnn_prob:.4f}  → {'⚠ IMPAIRED' if cnn_pred else '✓ HEALTHY'}")
        print(f"    Ensemble Probability  : {ensemble_prob:.4f}")

        print(f"\n  {risk_color} SCREENING RESULT : {risk_level}")
        print(f"     {advice}")

    print(f"\n{'═'*60}")
    print("  Demo complete. Both models loaded and functioning.")
    print("═"*60)

# ──────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("[INFO] Loading models...")
    scaler, svm_model, cnn_model = load_models()
    print("[OK]  All models loaded successfully.")
    run_demo(scaler, svm_model, cnn_model)
