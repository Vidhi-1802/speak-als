# =============================================================
# Speak-ALS Project — Phase 3b: CNN Model (1D on features)
# =============================================================
# A 1D-CNN that treats the 22 acoustic features as a sequence.
# This is more powerful than SVM and shows deep learning skill.
# Later we can extend this to 2D CNN on spectrograms.
# =============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import json
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, classification_report
)
import seaborn as sns

# TensorFlow / Keras
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'   # Suppress TF info messages
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, callbacks

BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
DATA_DIR      = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR    = os.path.join(BASE_DIR, "models")
PLOTS_DIR     = os.path.join(BASE_DIR, "plots")
RESULTS_DIR   = os.path.join(BASE_DIR, "results")

print(f"[INFO] TensorFlow version: {tf.__version__}")
tf.random.set_seed(42)
np.random.seed(42)

# ──────────────────────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────────────────────
def load_data():
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    X_test  = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    y_test  = np.load(os.path.join(DATA_DIR, "y_test.npy"))

    # Reshape for 1D CNN: (samples, features, 1)
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)

    print(f"[OK] CNN input shape: {X_train_cnn.shape}")
    return X_train_cnn, X_test_cnn, y_train, y_test

# ──────────────────────────────────────────────────────────────
# Build CNN model
# ──────────────────────────────────────────────────────────────
def build_model(input_shape):
    """
    Architecture:
      Conv1D(64) → BatchNorm → MaxPool
      Conv1D(128) → BatchNorm → MaxPool
      Conv1D(64) → BatchNorm
      GlobalAvgPool
      Dense(64) → Dropout(0.4)
      Dense(1, sigmoid) → binary output
    """
    model = keras.Sequential([
        # Input
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Block 2
        layers.Conv1D(128, kernel_size=3, padding='same', activation='relu',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(pool_size=2),
        layers.Dropout(0.2),

        # Block 3
        layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
        layers.BatchNormalization(),

        # Pooling → Dense
        layers.GlobalAveragePooling1D(),
        layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.4),
        layers.Dense(1, activation='sigmoid')   # Binary classification
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.AUC(name='auc'),
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall')]
    )
    return model

# ──────────────────────────────────────────────────────────────
# Train
# ──────────────────────────────────────────────────────────────
def train_model(model, X_train, y_train):
    # Callbacks
    early_stop = callbacks.EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True, verbose=1
    )
    reduce_lr = callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1
    )
    print("[INFO] Training CNN model...")
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_split=0.2,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    print("[OK] Training complete.")
    return history

# ──────────────────────────────────────────────────────────────
# Plots
# ──────────────────────────────────────────────────────────────
def plot_training_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle('CNN Training History', fontsize=13, fontweight='bold')

    # Loss
    axes[0].plot(history.history['loss'],     label='Train Loss', color='#3498db', lw=2)
    axes[0].plot(history.history['val_loss'], label='Val Loss',   color='#e74c3c', lw=2, linestyle='--')
    axes[0].set_title('Loss over Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Cross-Entropy')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    axes[1].plot(history.history['auc'],     label='Train AUC', color='#2ecc71', lw=2)
    axes[1].plot(history.history['val_auc'], label='Val AUC',   color='#e67e22', lw=2, linestyle='--')
    axes[1].set_title('AUC over Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "cnn_training_history.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Training history plot saved → {path}")

def plot_confusion_matrix(y_test, y_pred, model_name="CNN"):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Reds',
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

def plot_roc_curve(y_test, y_prob, model_name="CNN"):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='#9b59b6', lw=2, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#9b59b6')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title(f'{model_name} — ROC Curve', fontsize=13, fontweight='bold')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, f"roc_curve_{model_name.lower()}.png")
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] ROC curve saved → {path}")

# ──────────────────────────────────────────────────────────────
# Evaluate
# ──────────────────────────────────────────────────────────────
def evaluate(model, X_test, y_test, threshold=0.5):
    y_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_prob >= threshold).astype(int)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*50}")
    print(f"  CNN — TEST SET RESULTS")
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
# MAIN
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\n" + "="*55)
    print("  SPEAK-ALS — CNN MODEL TRAINING")
    print("="*55 + "\n")

    X_train, X_test, y_train, y_test = load_data()

    # Build and summarize model
    model = build_model(input_shape=(X_train.shape[1], 1))
    print("\n[MODEL ARCHITECTURE]")
    model.summary()

    # Train
    history = train_model(model, X_train, y_train)
    plot_training_history(history)

    # Evaluate
    y_pred, y_prob, metrics = evaluate(model, X_test, y_test)
    plot_confusion_matrix(y_test, y_pred, "CNN")
    plot_roc_curve(y_test, y_prob, "CNN")

    # Save model + metrics
    model.save(os.path.join(MODELS_DIR, "cnn_final.h5"))
    print(f"[OK] CNN model saved → {MODELS_DIR}/cnn_final.h5")

    metrics_path = os.path.join(RESULTS_DIR, "cnn_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({k: round(float(v), 4) for k, v in metrics.items()}, f, indent=2)
    print(f"[OK] Metrics saved → {metrics_path}")

    print("\n[NEXT STEP] Run: python 05_compare_models.py")
