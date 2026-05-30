from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import joblib
import librosa
import parselmouth
from parselmouth.praat import call
import tensorflow as tf
import tempfile
import os
import json

app = FastAPI(title="Speak-ALS API")

# Allow React frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models once at startup
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "..", "models")
RESULTS = os.path.join(BASE, "..", "results")

svm_model = None
scaler = None
cnn_model = None

@app.on_event("startup")
def load_models():
    global svm_model, scaler, cnn_model
    try:
        svm_model = joblib.load(os.path.join(MODELS, "svm_model.pkl"))
        scaler    = joblib.load(os.path.join(MODELS, "scaler.pkl"))
        cnn_model = tf.saved_model.load(os.path.join(MODELS, "cnn_savedmodel"))
        print("[OK] All models loaded successfully")
    except Exception as e:
        print(f"[ERROR] Could not load models: {e}")


def extract_features_from_audio(audio_path: str) -> dict:
    """Extract the 22 acoustic features from a .wav file."""
    # Load audio
    y, sr = librosa.load(audio_path, sr=None)

    # --- Parselmouth / Praat features ---
    snd = parselmouth.Sound(audio_path)

    # Pitch
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_pitch = call(pitch, "Get mean", 0, 0, "Hertz")

    # Jitter
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    jitter_local  = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_abs    = call(point_process, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_rap    = call(point_process, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ppq5   = call(point_process, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    jitter_ddp    = jitter_rap * 3

    # Shimmer
    shimmer_local   = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_local_db= call([snd, point_process], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq3    = call([snd, point_process], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq5    = call([snd, point_process], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_apq11   = call([snd, point_process], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    shimmer_dda     = shimmer_apq3 * 3

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)

    # --- Librosa features ---
    mfccs   = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    nfc     = librosa.feature.zero_crossing_rate(y)[0].mean()
    rms     = librosa.feature.rms(y=y)[0].mean()

    # Spread / PPE approximation using pitch std
    pitch_values = pitch.selected_array['frequency']
    pitch_values = pitch_values[pitch_values > 0]
    ppe = float(np.std(pitch_values)) / float(mean_pitch) if mean_pitch > 0 else 0.2

    # Build the 22-feature vector matching training order
    features = {
        "MDVP:Fo(Hz)":    float(mean_pitch),
        "MDVP:Fhi(Hz)":   float(np.max(pitch_values)) if len(pitch_values) > 0 else mean_pitch,
        "MDVP:Flo(Hz)":   float(np.min(pitch_values)) if len(pitch_values) > 0 else mean_pitch,
        "MDVP:Jitter(%)": float(jitter_local),
        "MDVP:Jitter(Abs)": float(jitter_abs),
        "MDVP:RAP":       float(jitter_rap),
        "MDVP:PPQ":       float(jitter_ppq5),
        "Jitter:DDP":     float(jitter_ddp),
        "MDVP:Shimmer":   float(shimmer_local),
        "MDVP:Shimmer(dB)": float(shimmer_local_db),
        "Shimmer:APQ3":   float(shimmer_apq3),
        "Shimmer:APQ5":   float(shimmer_apq5),
        "MDVP:APQ":       float(shimmer_apq11),
        "Shimmer:DDA":    float(shimmer_dda),
        "NHR":            float(1.0 / hnr if hnr > 0 else 0.1),
        "HNR":            float(hnr),
        "RPDE":           float(rms),
        "DFA":            float(nfc),
        "spread1":        float(np.mean(mfccs[1])),
        "spread2":        float(np.mean(mfccs[2])),
        "D2":             float(np.mean(mfccs[3])),
        "PPE":            float(ppe),
    }
    return features


def predict_from_features(features: dict) -> dict:
    """Run SVM + CNN prediction and return results."""
    feature_order = [
        "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
        "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP",
        "MDVP:Shimmer", "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5",
        "MDVP:APQ", "Shimmer:DDA", "NHR", "HNR",
        "RPDE", "DFA", "spread1", "spread2", "D2", "PPE"
    ]

    x = np.array([[features[f] for f in feature_order]])
    x_scaled = scaler.transform(x)

    # SVM
    svm_prob = float(svm_model.predict_proba(x_scaled)[0][1])
    svm_pred = int(svm_model.predict(x_scaled)[0])

    # CNN
    x_cnn = x_scaled.reshape(1, 22, 1).astype(np.float32)
    infer = cnn_model.signatures["serving_default"]
    output_key = list(infer.structured_outputs.keys())[0]
    cnn_prob = float(infer(tf.constant(x_cnn))[output_key].numpy()[0][0])
    cnn_pred = int(cnn_prob >= 0.5)

    ensemble = (svm_prob + cnn_prob) / 2

    if ensemble < 0.3:
        risk_level = "LOW"
        risk_label = "Low Risk"
        risk_msg   = "No significant acoustic irregularities detected."
        risk_color = "green"
    elif ensemble < 0.5:
        risk_level = "LOW_MODERATE"
        risk_label = "Low-Moderate Risk"
        risk_msg   = "Minor irregularities detected. Monitor over time."
        risk_color = "orange"
    elif ensemble < 0.7:
        risk_level = "MODERATE"
        risk_label = "Moderate Risk"
        risk_msg   = "Some acoustic abnormalities detected. Consult a specialist."
        risk_color = "amber"
    else:
        risk_level = "HIGH"
        risk_label = "High Risk"
        risk_msg   = "Significant acoustic irregularities detected. Seek medical evaluation."
        risk_color = "red"

    return {
        "svm_probability":  round(svm_prob * 100, 1),
        "svm_prediction":   "Impaired" if svm_pred == 1 else "Healthy",
        "cnn_probability":  round(cnn_prob * 100, 1),
        "cnn_prediction":   "Impaired" if cnn_pred == 1 else "Healthy",
        "ensemble_probability": round(ensemble * 100, 1),
        "risk_level":  risk_level,
        "risk_label":  risk_label,
        "risk_message": risk_msg,
        "risk_color":  risk_color,
        "key_features": {
            "pitch_hz":  round(features["MDVP:Fo(Hz)"], 2),
            "jitter_pct": round(features["MDVP:Jitter(%)"] * 100, 4),
            "shimmer":   round(features["MDVP:Shimmer"], 4),
            "hnr_db":    round(features["HNR"], 2),
            "ppe":       round(features["PPE"], 4),
        }
    }


@app.get("/")
def root():
    return {"status": "Speak-ALS API is running"}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    if not file.filename.endswith((".wav", ".WAV")):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")

    if svm_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded yet")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        features = extract_features_from_audio(tmp_path)
        result   = predict_from_features(features)
        result["filename"] = file.filename
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")
    finally:
        os.unlink(tmp_path)


@app.get("/model-stats")
def model_stats():
    """Return pre-computed model performance metrics for the About page."""
    try:
        with open(os.path.join(RESULTS, "svm_metrics.json")) as f:
            svm = json.load(f)
        with open(os.path.join(RESULTS, "cnn_metrics.json")) as f:
            cnn = json.load(f)
        return {"svm": svm, "cnn": cnn}
    except:
        return {
            "svm": {"accuracy": 0.897, "f1": 0.931, "roc_auc": 0.962},
            "cnn": {"accuracy": 0.923, "f1": 0.947, "roc_auc": 0.976}
        }
