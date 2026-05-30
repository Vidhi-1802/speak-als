import { useEffect, useState } from 'react'

const riskConfig = {
  LOW:          { color: '#b8ddd4', bg: 'var(--mint-light)',  border: 'var(--mint)',    emoji: '🟢', label: 'Low Risk' },
  LOW_MODERATE: { color: '#e8a87c', bg: 'var(--peach-light)', border: 'var(--peach)',   emoji: '🟠', label: 'Low-Moderate Risk' },
  MODERATE:     { color: '#d4a843', bg: '#fff8e0',            border: '#f0d080',        emoji: '🟡', label: 'Moderate Risk' },
  HIGH:         { color: '#e07070', bg: '#fff0f0',            border: '#f2c4c4',        emoji: '🔴', label: 'High Risk' },
}

function ProgressBar({ value, color, label }) {
  const [width, setWidth] = useState(0)
  useEffect(() => { setTimeout(() => setWidth(value), 300) }, [value])
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '6px' }}>
        <span style={{ fontSize: '0.85rem', color: 'var(--text-mid)' }}>{label}</span>
        <span style={{ fontSize: '0.9rem', fontWeight: 600, color: 'var(--text-dark)' }}>{value}%</span>
      </div>
      <div style={{ background: 'var(--cream)', borderRadius: 999, height: 10, overflow: 'hidden' }}>
        <div style={{
          width: `${width}%`, height: '100%',
          background: color, borderRadius: 999,
          transition: 'width 1.2s cubic-bezier(0.4,0,0.2,1)',
        }} />
      </div>
    </div>
  )
}

function FeatureCard({ label, value, unit, color }) {
  return (
    <div className="card" style={{ background: color, boxShadow: 'none', padding: '1.2rem', textAlign: 'center' }}>
      <div style={{ fontFamily: 'Cormorant Garamond, serif', fontSize: '1.8rem', fontWeight: 600 }}>{value}</div>
      <div style={{ fontSize: '0.72rem', color: 'var(--text-soft)', marginTop: 2 }}>{unit}</div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-mid)', marginTop: 4, fontWeight: 500 }}>{label}</div>
    </div>
  )
}

export default function Results({ data, navigate }) {
  const [visible, setVisible] = useState(false)
  useEffect(() => { setTimeout(() => setVisible(true), 100) }, [])

  if (!data) return (
    <div style={{ textAlign: 'center', padding: '5rem 2rem' }}>
      <p style={{ color: 'var(--text-soft)' }}>No results yet.</p>
      <button onClick={() => navigate('analyze')} style={{
        marginTop: '1rem', background: 'var(--lav-light)',
        border: 'none', padding: '10px 24px', borderRadius: 999,
        cursor: 'pointer', color: 'var(--text-mid)',
      }}>Go to Analyze</button>
    </div>
  )

  const risk = riskConfig[data.risk_level] || riskConfig.LOW
  const f = data.key_features || {}

  return (
    <div className="page-enter" style={{ maxWidth: 820, margin: '0 auto', padding: '3rem 2rem 5rem' }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
        <div className="pill" style={{ background: 'var(--mint-light)', color: 'var(--text-mid)', marginBottom: '1rem' }}>
          ✅ Analysis Complete
        </div>
        <h1 style={{ fontSize: '2.6rem', fontWeight: 300 }}>Your Results</h1>
        {data.filename && (
          <p style={{ color: 'var(--text-soft)', fontSize: '0.85rem', marginTop: '6px' }}>
            File: {data.filename}
          </p>
        )}
      </div>

      {/* Risk banner */}
      <div style={{
        background: risk.bg,
        border: `2px solid ${risk.border}`,
        borderRadius: 'var(--radius-lg)',
        padding: '2rem', textAlign: 'center',
        marginBottom: '2rem',
        animation: 'fadeUp 0.6s ease',
      }}>
        <div style={{ fontSize: '2.5rem', marginBottom: '0.5rem' }}>{risk.emoji}</div>
        <div style={{ fontFamily: 'Cormorant Garamond, serif', fontSize: '2rem', fontWeight: 600, marginBottom: '0.5rem' }}>
          {risk.label}
        </div>
        <div style={{ fontSize: '1.5rem', fontWeight: 700, color: risk.color, marginBottom: '0.5rem' }}>
          {data.ensemble_probability}% Risk Probability
        </div>
        <p style={{ color: 'var(--text-mid)', fontSize: '0.95rem', maxWidth: 500, margin: '0 auto' }}>
          {data.risk_message}
        </p>
      </div>

      {/* Model predictions */}
      <div className="card" style={{ marginBottom: '1.5rem' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 600, marginBottom: '1.5rem' }}>Model Predictions</h2>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.2rem' }}>
          <ProgressBar
            value={data.svm_probability}
            color="linear-gradient(90deg, #c9c0e3, #a08cc0)"
            label={`SVM Model → ${data.svm_prediction}`}
          />
          <ProgressBar
            value={data.cnn_probability}
            color="linear-gradient(90deg, #f2c4ce, #e08090)"
            label={`CNN Model → ${data.cnn_prediction}`}
          />
          <ProgressBar
            value={data.ensemble_probability}
            color="linear-gradient(90deg, #b8ddd4, #80bfb5)"
            label="Ensemble (Combined)"
          />
        </div>
      </div>

      {/* Acoustic features */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.4rem', fontWeight: 600, marginBottom: '1.2rem' }}>Extracted Acoustic Features</h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.8rem' }}>
          <FeatureCard label="Pitch"   value={f.pitch_hz}    unit="Hz"  color="var(--lav-light)" />
          <FeatureCard label="Jitter"  value={f.jitter_pct}  unit="%"   color="var(--blush-light)" />
          <FeatureCard label="Shimmer" value={f.shimmer}     unit="—"   color="var(--mint-light)" />
          <FeatureCard label="HNR"     value={f.hnr_db}      unit="dB"  color="var(--peach-light)" />
          <FeatureCard label="PPE"     value={f.ppe}         unit="—"   color="var(--lav-light)" />
        </div>
      </div>

      {/* What these mean */}
      <div className="card" style={{ background: 'var(--lav-light)', boxShadow: 'none', border: '1.5px solid var(--lavender)', marginBottom: '2rem' }}>
        <h3 style={{ fontSize: '1.1rem', marginBottom: '0.75rem' }}>📖 What do these features mean?</h3>
        <div style={{ fontSize: '0.85rem', color: 'var(--text-mid)', lineHeight: 1.7 }}>
          <strong>Jitter</strong> — variation in pitch frequency. Higher values suggest vocal instability.<br />
          <strong>Shimmer</strong> — variation in amplitude. Higher values indicate reduced voice control.<br />
          <strong>HNR</strong> — ratio of harmonic to noise. Lower HNR = more noise in the voice signal.<br />
          <strong>PPE</strong> — pitch period entropy. Measures regularity of pitch patterns over time.
        </div>
      </div>

      {/* Disclaimer */}
      <div style={{
        textAlign: 'center', padding: '1.5rem',
        background: '#fff8f0', borderRadius: 'var(--radius-md)',
        border: '1.5px solid var(--peach)', marginBottom: '2rem',
        fontSize: '0.82rem', color: 'var(--text-mid)',
      }}>
        ⚠ <strong>Important:</strong> This is a research prototype and is NOT a substitute for medical diagnosis.
        Please consult a qualified neurologist or speech-language pathologist for any health concerns.
      </div>

      {/* Actions */}
      <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
        <button onClick={() => navigate('analyze')} style={{
          background: 'linear-gradient(135deg, #c9c0e3, #f2c4ce)',
          color: 'var(--text-dark)', padding: '12px 28px',
          borderRadius: 999, fontWeight: 600, fontSize: '0.95rem',
          boxShadow: '0 4px 16px rgba(201,192,227,0.35)',
        }}>Analyze Another Recording</button>

        <button onClick={() => navigate('how-it-works')} style={{
          background: 'transparent', border: '1.5px solid var(--lavender)',
          color: 'var(--text-mid)', padding: '12px 28px',
          borderRadius: 999, fontWeight: 500, fontSize: '0.95rem',
        }}>Learn How It Works</button>
      </div>
    </div>
  )
}
