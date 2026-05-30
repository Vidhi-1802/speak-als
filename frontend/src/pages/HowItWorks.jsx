const pipeline = [
  {
    step: '01',
    title: 'Voice Recording',
    desc: 'The user records a sustained vowel sound ("ahhh") for 3–5 seconds. This sustained phonation isolates vocal cord behavior from linguistic complexity.',
    color: 'var(--lav-light)', accent: 'var(--lavender)',
    icon: '🎙',
  },
  {
    step: '02',
    title: 'Audio Preprocessing',
    desc: 'The .wav file is loaded using librosa. The signal is normalized and prepared for feature extraction using Praat-compatible algorithms via parselmouth.',
    color: 'var(--blush-light)', accent: 'var(--blush)',
    icon: '⚙️',
  },
  {
    step: '03',
    title: 'Acoustic Feature Extraction',
    desc: '22 clinically validated acoustic features are extracted — including fundamental frequency, jitter (pitch variability), shimmer (amplitude variability), HNR (harmonic-to-noise ratio), and PPE (pitch period entropy).',
    color: 'var(--mint-light)', accent: 'var(--mint)',
    icon: '📐',
  },
  {
    step: '04',
    title: 'SVM Classification',
    desc: 'A Support Vector Machine with RBF kernel (C=10, gamma=scale) trained on the UCI Parkinson\'s Voice Dataset. Achieves 89.7% accuracy and 96.2% ROC-AUC on test data.',
    color: 'var(--peach-light)', accent: 'var(--peach)',
    icon: '🔷',
  },
  {
    step: '05',
    title: '1D Convolutional Neural Network',
    desc: 'A 3-layer 1D CNN with BatchNormalization, MaxPooling, and Dropout layers. Trained for 80 epochs with early stopping. Achieves 92.3% accuracy and 97.6% ROC-AUC.',
    color: 'var(--lav-light)', accent: 'var(--lavender)',
    icon: '🧠',
  },
  {
    step: '06',
    title: 'Ensemble Decision',
    desc: 'The final risk probability is the average of SVM and CNN outputs. Risk is categorized as Low (<30%), Low-Moderate (30–50%), Moderate (50–70%), or High (>70%).',
    color: 'var(--mint-light)', accent: 'var(--mint)',
    icon: '⚖️',
  },
]

const features = [
  { name: 'MDVP:Fo (Hz)',     desc: 'Average vocal fundamental frequency' },
  { name: 'MDVP:Jitter (%)',  desc: 'Percentage of variation in pitch' },
  { name: 'MDVP:Shimmer',     desc: 'Variation in amplitude of speech signal' },
  { name: 'HNR',              desc: 'Harmonic-to-noise ratio in dB' },
  { name: 'RPDE',             desc: 'Recurrence period density entropy' },
  { name: 'DFA',              desc: 'Detrended fluctuation analysis' },
  { name: 'PPE',              desc: 'Pitch period entropy' },
  { name: 'Jitter:DDP',       desc: '3× jitter RAP — widely used dysarthria marker' },
  { name: 'Shimmer:APQ11',    desc: '11-point amplitude perturbation quotient' },
  { name: 'NHR',              desc: 'Noise-to-harmonics ratio' },
]

export default function HowItWorks() {
  return (
    <div className="page-enter" style={{ maxWidth: 900, margin: '0 auto', padding: '3rem 2rem 5rem' }}>

      <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
        <div className="pill" style={{ background: 'var(--lav-light)', color: 'var(--text-mid)', marginBottom: '1rem' }}>
          Technical Overview
        </div>
        <h1 style={{ fontSize: '2.8rem', fontWeight: 300, marginBottom: '0.75rem' }}>
          How Speak-ALS Works
        </h1>
        <p style={{ color: 'var(--text-mid)', maxWidth: 560, margin: '0 auto', lineHeight: 1.7, fontSize: '0.95rem' }}>
          A six-stage clinical acoustic analysis pipeline — from raw audio to risk indication.
        </p>
      </div>

      {/* Pipeline */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem', marginBottom: '4rem', position: 'relative' }}>
        {pipeline.map((p, i) => (
          <div key={i} style={{ display: 'flex', gap: '1.5rem', alignItems: 'flex-start' }}>
            {/* Step number + connector */}
            <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', flexShrink: 0 }}>
              <div style={{
                width: 52, height: 52, borderRadius: '50%',
                background: p.color, border: `2px solid ${p.accent}`,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontFamily: 'Cormorant Garamond, serif', fontSize: '1rem', fontWeight: 600,
                color: 'var(--text-dark)',
              }}>{p.step}</div>
              {i < pipeline.length - 1 && (
                <div style={{ width: 2, flex: 1, minHeight: 24, background: 'rgba(201,192,227,0.3)', marginTop: 4 }} />
              )}
            </div>
            {/* Content */}
            <div className="card" style={{ flex: 1, marginBottom: 0, transition: 'transform 0.2s' }}
              onMouseEnter={e => e.currentTarget.style.transform='translateX(4px)'}
              onMouseLeave={e => e.currentTarget.style.transform='translateX(0)'}
            >
              <div style={{ display: 'flex', gap: '10px', alignItems: 'center', marginBottom: '0.5rem' }}>
                <span style={{ fontSize: '1.3rem' }}>{p.icon}</span>
                <h3 style={{ fontSize: '1.2rem', fontWeight: 600 }}>{p.title}</h3>
              </div>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-mid)', lineHeight: 1.7 }}>{p.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Feature table */}
      <div className="card" style={{ marginBottom: '3rem' }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 300, marginBottom: '0.5rem' }}>Key Acoustic Features</h2>
        <p style={{ color: 'var(--text-soft)', fontSize: '0.85rem', marginBottom: '1.5rem' }}>
          10 of the 22 features extracted from each voice sample
        </p>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '0' }}>
          {features.map((f, i) => (
            <div key={i} style={{
              display: 'grid', gridTemplateColumns: '1fr 2fr',
              padding: '0.75rem 0',
              borderBottom: i < features.length - 1 ? '1px solid rgba(201,192,227,0.2)' : 'none',
            }}>
              <code style={{
                fontSize: '0.82rem', fontFamily: 'monospace',
                background: 'var(--lav-light)', padding: '2px 8px',
                borderRadius: 6, alignSelf: 'start',
                color: 'var(--text-dark)', width: 'fit-content',
              }}>{f.name}</code>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-mid)', lineHeight: 1.5 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Model comparison */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.2rem', marginBottom: '3rem' }}>
        <div className="card" style={{ background: 'var(--lav-light)', boxShadow: 'none', border: '1.5px solid var(--lavender)' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '0.75rem' }}>🔷 Support Vector Machine</h3>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-mid)', lineHeight: 1.8 }}>
            <div>Kernel: RBF (C=10, γ=scale)</div>
            <div>Tuning: GridSearchCV (5-fold)</div>
            <div>Accuracy: <strong>89.7%</strong></div>
            <div>F1-Score: <strong>93.1%</strong></div>
            <div>ROC-AUC: <strong>96.2%</strong></div>
          </div>
        </div>
        <div className="card" style={{ background: 'var(--blush-light)', boxShadow: 'none', border: '1.5px solid var(--blush)' }}>
          <h3 style={{ fontSize: '1.2rem', marginBottom: '0.75rem' }}>🧠 1D Convolutional Neural Network</h3>
          <div style={{ fontSize: '0.84rem', color: 'var(--text-mid)', lineHeight: 1.8 }}>
            <div>Layers: Conv1D × 3, Dense × 2</div>
            <div>Training: 80 epochs, early stopping</div>
            <div>Accuracy: <strong>92.3%</strong></div>
            <div>F1-Score: <strong>94.7%</strong></div>
            <div>ROC-AUC: <strong>97.6%</strong></div>
          </div>
        </div>
      </div>

      {/* Dataset */}
      <div className="card" style={{ background: 'var(--mint-light)', boxShadow: 'none', border: '1.5px solid var(--mint)' }}>
        <h3 style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>📂 Dataset</h3>
        <p style={{ fontSize: '0.88rem', color: 'var(--text-mid)', lineHeight: 1.7 }}>
          Trained on the <strong>UCI Parkinson's Voice Dataset</strong> — 195 voice recordings from 31 subjects (23 with Parkinson's/motor speech disorder, 8 healthy controls).
          The acoustic features are clinically equivalent to those used in ALS speech screening research.
          The dataset is publicly available from the UCI Machine Learning Repository.
        </p>
      </div>
    </div>
  )
}
