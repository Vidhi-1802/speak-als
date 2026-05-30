import streamlit as st
import numpy as np
import joblib
import os
import tempfile
import warnings
warnings.filterwarnings('ignore')

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Speak-ALS · Early Detection",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ── Custom CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@300;400;600;700;800&family=Playfair+Display:wght@400;700&display=swap');

/* Reset & base */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #fdf6f0 !important;
    font-family: 'Nunito', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #fdf6f0 0%, #f3eefe 50%, #e8f6f0 100%) !important;
    min-height: 100vh;
}

/* Hide default streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* Hero section */
.hero {
    text-align: center;
    padding: 3rem 1rem 2rem;
    animation: fadeDown 0.8s ease both;
}
.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #f9c6d0, #c8b6f7);
    color: #5a3e6b;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: clamp(2.2rem, 5vw, 3.2rem);
    font-weight: 700;
    background: linear-gradient(135deg, #d4689a, #7c5cbf, #4aab8c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin: 0 0 0.75rem;
}
.hero-sub {
    color: #8a7a9b;
    font-size: 1.05rem;
    font-weight: 400;
    max-width: 480px;
    margin: 0 auto 0.5rem;
    line-height: 1.6;
}

/* Step cards */
.step-wrap {
    display: flex;
    gap: 0.6rem;
    justify-content: center;
    margin: 1.8rem 0 0.5rem;
    flex-wrap: wrap;
}
.step-pill {
    display: flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 0.9rem;
    border-radius: 999px;
    font-size: 0.8rem;
    font-weight: 700;
    transition: transform 0.2s;
}
.step-pill:hover { transform: translateY(-2px); }
.step-1 { background: #fde8ee; color: #c1547a; border: 1.5px solid #f7c0d0; }
.step-2 { background: #ede8fe; color: #6b49c8; border: 1.5px solid #cfc0f7; }
.step-3 { background: #e4f7f0; color: #2e9e72; border: 1.5px solid #a8e5cf; }
.step-num {
    width: 18px; height: 18px;
    border-radius: 50%;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 800;
}
.step-1 .step-num { background: #f9b8cc; }
.step-2 .step-num { background: #c9b8f9; }
.step-3 .step-num { background: #96dfc4; }

/* Upload card */
.upload-card {
    background: white;
    border-radius: 24px;
    padding: 2rem 2rem 1.5rem;
    box-shadow: 0 4px 30px rgba(180,150,220,0.10), 0 1px 4px rgba(0,0,0,0.04);
    border: 1.5px solid #ede8fe;
    margin: 1.5rem 0;
    animation: fadeUp 0.7s ease 0.2s both;
}
.card-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #b89fd4;
    margin-bottom: 0.5rem;
}
.card-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #3d2e5e;
    margin-bottom: 0.3rem;
}
.card-desc {
    color: #9e8fb5;
    font-size: 0.92rem;
    margin-bottom: 1.2rem;
    line-height: 1.5;
}

/* Tip box */
.tip-box {
    background: linear-gradient(135deg, #fef3f7, #f3effe);
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    font-size: 0.85rem;
    color: #7a6090;
    border-left: 3px solid #d4a0c8;
    margin-top: 1rem;
    line-height: 1.5;
}

/* Metric cards */
.metrics-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 0.9rem;
    margin: 1rem 0;
}
.metric-card {
    background: white;
    border-radius: 18px;
    padding: 1.1rem 1.2rem;
    border: 1.5px solid #ede8fe;
    box-shadow: 0 2px 12px rgba(180,150,220,0.08);
}
.metric-label {
    font-size: 0.72rem;
    font-weight: 800;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #b89fd4;
    margin-bottom: 0.25rem;
}
.metric-value {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 700;
    color: #3d2e5e;
    line-height: 1.1;
}
.metric-unit {
    font-size: 0.75rem;
    color: #b89fd4;
    font-weight: 600;
    margin-left: 2px;
}

/* Model comparison */
.model-row {
    background: white;
    border-radius: 18px;
    padding: 1.2rem 1.4rem;
    border: 1.5px solid #ede8fe;
    box-shadow: 0 2px 12px rgba(180,150,220,0.08);
    margin-bottom: 0.8rem;
}
.model-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.7rem;
}
.model-name {
    font-weight: 800;
    font-size: 0.95rem;
    color: #3d2e5e;
}
.model-badge {
    font-size: 0.72rem;
    font-weight: 700;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
}
.badge-healthy { background: #e4f7f0; color: #2e9e72; }
.badge-risk    { background: #fde8ee; color: #c1547a; }

.prob-bar-bg {
    background: #f3effe;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-bottom: 0.35rem;
}
.prob-bar-fill {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}
.bar-svm  { background: linear-gradient(90deg, #c9b8f9, #9b7fe8); }
.bar-cnn  { background: linear-gradient(90deg, #f9b8cc, #d4689a); }
.prob-pct {
    font-size: 0.8rem;
    font-weight: 700;
    color: #8a7a9b;
    text-align: right;
}

/* Result banner */
.result-banner {
    border-radius: 22px;
    padding: 1.6rem 1.8rem;
    text-align: center;
    margin: 1.2rem 0;
    animation: pulse 2s ease infinite;
}
.result-low     { background: linear-gradient(135deg, #e4f7f0, #d0f5e8); border: 2px solid #96dfc4; }
.result-medium  { background: linear-gradient(135deg, #fef9e4, #fef0c0); border: 2px solid #f7dc7a; }
.result-high    { background: linear-gradient(135deg, #fde8ee, #fcd0db); border: 2px solid #f9b8cc; }
.result-emoji { font-size: 2.5rem; margin-bottom: 0.4rem; }
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}
.result-low .result-title    { color: #1e7d55; }
.result-medium .result-title { color: #8a6a00; }
.result-high .result-title   { color: #a83258; }
.result-desc {
    font-size: 0.9rem;
    font-weight: 600;
    line-height: 1.5;
}
.result-low .result-desc    { color: #2e9e72; }
.result-medium .result-desc { color: #b08a00; }
.result-high .result-desc   { color: #c1547a; }

/* Disclaimer */
.disclaimer {
    background: #fdf0f5;
    border: 1.5px dashed #f9b8cc;
    border-radius: 14px;
    padding: 0.9rem 1.1rem;
    font-size: 0.8rem;
    color: #a07090;
    text-align: center;
    margin-top: 1.5rem;
    line-height: 1.5;
}

/* Divider */
.soft-divider {
    border: none;
    border-top: 1.5px solid #ede8fe;
    margin: 1.5rem 0;
}

/* Animations */
@keyframes fadeDown {
    from { opacity:0; transform:translateY(-18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(18px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse {
    0%,100% { box-shadow: 0 4px 20px rgba(180,150,220,0.12); }
    50%      { box-shadow: 0 4px 32px rgba(180,150,220,0.28); }
}

/* Streamlit overrides */
[data-testid="stFileUploader"] {
    background: #fdf6ff;
    border: 2px dashed #cfc0f7 !important;
    border-radius: 16px !important;
    padding: 0.5rem !important;
}
.stButton > button {
    background: linear-gradient(135deg, #c9b8f9, #f9b8cc) !important;
    color: #3d2e5e !important;
    font-family: 'Nunito', sans-serif !important;
    font-weight: 800 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 14px !important;
    padding: 0.7rem 2.2rem !important;
    width: 100% !important;
    transition: transform 0.2s, box-shadow 0.2s !important;
    box-shadow: 0 4px 18px rgba(180,150,220,0.25) !important;
    letter-spacing: 0.02em !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(180,150,220,0.38) !important;
}
</style>
""", unsafe_allow_html=True)


# ── Hero ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🎙️ Final Year Project · AIML</div>
    <div class="hero-title">Speak-ALS</div>
    <div class="hero-sub">Early detection of ALS-related speech impairment using dual AI models — SVM & CNN — trained on acoustic voice biomarkers.</div>
    <div class="step-wrap">
        <div class="step-pill step-1"><span class="step-num">1</span> Upload Audio</div>
        <div class="step-pill step-2"><span class="step-num">2</span> Analyze Features</div>
        <div class="step-pill step-3"><span class="step-num">3</span> View Results</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Load Models ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    import tensorflow as tf
    scaler    = joblib.load('models/scaler.pkl')
    svm_model = joblib.load('models/svm_model.pkl')
    cnn_model = tf.saved_model.load('models/cnn_savedmodel')
    return scaler, svm_model, cnn_model

try:
    scaler, svm_model, cnn_model = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    st.error(f"⚠️ Could not load models: {e}")


# ── Feature Extraction ──────────────────────────────────────────────────────────
def extract_features(audio_path):
    import librosa, parselmouth
    from parselmouth.praat import call

    y, sr = librosa.load(audio_path, sr=22050, mono=True)
    y, _ = librosa.effects.trim(y, top_db=20)

    # MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    # Pitch
    snd  = parselmouth.Sound(audio_path)
    pitch = call(snd, "To Pitch", 0.0, 75, 600)
    mean_f0 = call(pitch, "Get mean", 0, 0, "Hertz")
    if np.isnan(mean_f0): mean_f0 = 0.0

    # Jitter & Shimmer
    point_process = call(snd, "To PointProcess (periodic, cc)", 75, 600)
    try:
        jitter  = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        shimmer = call([snd, point_process], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except:
        jitter, shimmer = 0.0, 0.0

    # HNR
    harmonicity = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    if np.isnan(hnr): hnr = 0.0

    features = np.array([
        mean_f0, jitter, jitter, jitter, jitter,
        shimmer, shimmer, shimmer, shimmer, shimmer, shimmer,
        hnr,
        *mfcc_means[:10]
    ], dtype=np.float64)[:22]

    return features, {"pitch": mean_f0, "jitter": jitter, "shimmer": shimmer, "hnr": hnr}


# ── Step 1 — Upload ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-card">
    <div class="card-label">Step 01</div>
    <div class="card-title">Upload a Voice Sample</div>
    <div class="card-desc">Record yourself saying <strong>"ahhhhh"</strong> for 3–5 seconds and save as a <code>.wav</code> file. On Mac: open QuickTime → File → New Audio Recording.</div>
</div>
""", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["wav"], label_visibility="collapsed")

st.markdown("""
<div class="tip-box">
    💡 <strong>Recording tip:</strong> Sit in a quiet room, hold the mic 15–20 cm from your mouth, and sustain a steady "ahhh" tone for at least 3 seconds for the best results.
</div>
""", unsafe_allow_html=True)


# ── Step 2 & 3 — Analyze & Results ─────────────────────────────────────────────
if uploaded and models_ok:
    st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)

    st.markdown("""
    <div class="upload-card" style="border-color:#e4f7f0; animation-delay:0.1s;">
        <div class="card-label" style="color:#4aab8c;">Step 02</div>
        <div class="card-title">Acoustic Feature Analysis</div>
        <div class="card-desc">Extracting voice biomarkers — pitch, jitter, shimmer, HNR & MFCCs.</div>
    </div>
    """, unsafe_allow_html=True)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    with st.spinner("Analysing your voice sample…"):
        try:
            features_raw, display_feats = extract_features(tmp_path)
            features_scaled = scaler.transform(features_raw.reshape(1, -1))

            # SVM
            svm_prob = float(svm_model.predict_proba(features_scaled)[0][1])
            svm_pred = 1 if svm_prob >= 0.5 else 0

            # CNN
            import tensorflow as tf
            cnn_input = tf.constant(features_scaled.reshape(1, 22, 1), dtype=tf.float32)
            infer = cnn_model.signatures["serving_default"]
            output_key = list(infer(cnn_input).keys())[0]
            cnn_prob = float(infer(cnn_input)[output_key].numpy()[0][0])
            cnn_pred = 1 if cnn_prob >= 0.5 else 0

            ensemble = (svm_prob + cnn_prob) / 2
            os.unlink(tmp_path)

            # ── Feature cards ───────────────────────────────────────────────
            st.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-card">
                    <div class="metric-label">Fundamental Pitch</div>
                    <div class="metric-value">{display_feats['pitch']:.1f}<span class="metric-unit">Hz</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Jitter</div>
                    <div class="metric-value">{display_feats['jitter']*100:.3f}<span class="metric-unit">%</span></div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">Shimmer</div>
                    <div class="metric-value">{display_feats['shimmer']:.4f}</div>
                </div>
                <div class="metric-card">
                    <div class="metric-label">HNR</div>
                    <div class="metric-value">{display_feats['hnr']:.2f}<span class="metric-unit">dB</span></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Step 3 — Results ────────────────────────────────────────────
            st.markdown("<hr class='soft-divider'>", unsafe_allow_html=True)
            st.markdown("""
            <div class="upload-card" style="border-color:#fde8ee; animation-delay:0.2s;">
                <div class="card-label" style="color:#d4689a;">Step 03</div>
                <div class="card-title">Model Predictions</div>
                <div class="card-desc">Two independent AI models analyse your sample and vote on the result.</div>
            </div>
            """, unsafe_allow_html=True)

            svm_badge = 'badge-risk' if svm_pred else 'badge-healthy'
            svm_label = '⚠ Impaired' if svm_pred else '✓ Healthy'
            cnn_badge = 'badge-risk' if cnn_pred else 'badge-healthy'
            cnn_label = '⚠ Impaired' if cnn_pred else '✓ Healthy'

            st.markdown(f"""
            <div class="model-row">
                <div class="model-header">
                    <div class="model-name">🔷 SVM — Support Vector Machine</div>
                    <span class="model-badge {svm_badge}">{svm_label}</span>
                </div>
                <div class="prob-bar-bg"><div class="prob-bar-fill bar-svm" style="width:{svm_prob*100:.1f}%"></div></div>
                <div class="prob-pct">Risk probability: {svm_prob*100:.1f}%</div>
            </div>

            <div class="model-row">
                <div class="model-header">
                    <div class="model-name">🔶 CNN — Convolutional Neural Network</div>
                    <span class="model-badge {cnn_badge}">{cnn_label}</span>
                </div>
                <div class="prob-bar-bg"><div class="prob-bar-fill bar-cnn" style="width:{cnn_prob*100:.1f}%"></div></div>
                <div class="prob-pct">Risk probability: {cnn_prob*100:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            # ── Ensemble result ─────────────────────────────────────────────
            if ensemble < 0.35:
                cls, emoji, title, desc = "result-low", "🟢", "Low Risk", "No significant speech irregularities detected. Your voice biomarkers appear within normal range."
            elif ensemble < 0.60:
                cls, emoji, title, desc = "result-medium", "🟡", "Low–Moderate Risk", "Minor acoustic irregularities detected. Consider monitoring over time and consult a specialist if symptoms worsen."
            else:
                cls, emoji, title, desc = "result-high", "🔴", "Moderate–High Risk", "Notable speech abnormalities detected. Follow-up with a speech-language pathologist or neurologist is advised."

            st.markdown(f"""
            <div class="result-banner {cls}">
                <div class="result-emoji">{emoji}</div>
                <div class="result-title">{title}</div>
                <div class="result-desc">Ensemble score: {ensemble*100:.1f}% &nbsp;·&nbsp; {desc}</div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Analysis failed: {e}")
            st.info("Make sure parselmouth is installed: `pip3 install praat-parselmouth`")

elif uploaded and not models_ok:
    st.warning("Models not loaded. Please check your models/ folder.")

# ── Disclaimer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="disclaimer">
    ⚕️ <strong>Research Prototype Only.</strong> This tool is built for academic demonstration as part of a final year engineering project. It is <strong>not</strong> a certified medical diagnostic device and should not be used as a substitute for professional medical advice.
</div>
""", unsafe_allow_html=True)
