import { useEffect, useState } from 'react'

const stats = [
  { value: '97.6%', label: 'ROC-AUC Score', color: 'var(--lav-light)', accent: 'var(--lavender)' },
  { value: '92.3%', label: 'CNN Accuracy',  color: 'var(--blush-light)', accent: 'var(--blush)' },
  { value: '93.1%', label: 'Recall Rate',   color: 'var(--mint-light)', accent: 'var(--mint)' },
  { value: '~30s',  label: 'Analysis Time', color: 'var(--peach-light)', accent: 'var(--peach)' },
]

const features = [
  { icon: '🎙', title: 'Upload Voice', desc: 'Record a sustained "ahhh" sound and upload the .wav file directly in your browser.' },
  { icon: '🧠', title: 'Dual AI Models', desc: 'Your audio is analyzed by both SVM and a 1D Convolutional Neural Network simultaneously.' },
  { icon: '📊', title: 'Acoustic Features', desc: 'We extract 22 clinical features — jitter, shimmer, HNR, pitch, and more — just like a speech lab.' },
  { icon: '🩺', title: 'Risk Indication', desc: 'Get a color-coded risk level with ensemble probability from both models combined.' },
]

export default function Home({ navigate }) {
  const [visible, setVisible] = useState(false)
  useEffect(() => { setTimeout(() => setVisible(true), 100) }, [])

  return (
    <div className="page-enter" style={{ maxWidth: 1100, margin: '0 auto', padding: '0 2rem 4rem' }}>

      {/* Hero */}
      <section style={{
        textAlign: 'center',
        padding: '5rem 1rem 3rem',
        position: 'relative',
      }}>
        {/* Blob decorations */}
        <div style={{
          position: 'absolute', top: 20, left: '10%',
          width: 300, height: 300, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(201,192,227,0.25), transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none',
        }} />
        <div style={{
          position: 'absolute', top: 60, right: '10%',
          width: 250, height: 250, borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(242,196,206,0.25), transparent 70%)',
          filter: 'blur(40px)', pointerEvents: 'none',
        }} />

        <div className="pill" style={{ background: 'var(--lav-light)', color: 'var(--text-mid)', marginBottom: '1.5rem' }}>
          🎓 Final Year Engineering Project · AIML
        </div>

        <h1 style={{
          fontFamily: 'Cormorant Garamond, serif',
          fontSize: 'clamp(2.8rem, 6vw, 4.8rem)',
          fontWeight: 300,
          lineHeight: 1.1,
          color: 'var(--text-dark)',
          marginBottom: '1.5rem',
        }}>
          Early ALS Detection<br />
          <em style={{ color: '#a08cc0', fontStyle: 'italic' }}>Through Your Voice</em>
        </h1>

        <p style={{
          fontSize: '1.1rem', color: 'var(--text-mid)',
          maxWidth: 560, margin: '0 auto 2.5rem',
          lineHeight: 1.7, fontWeight: 300,
        }}>
          Speak-ALS uses machine learning to analyze subtle acoustic patterns in speech 
          that may indicate early motor neuron deterioration — non-invasively, in seconds.
        </p>

        <div style={{ display: 'flex', gap: '1rem', justifyContent: 'center', flexWrap: 'wrap' }}>
          <button
            onClick={() => navigate('analyze')}
            style={{
              background: 'linear-gradient(135deg, #c9c0e3 0%, #f2c4ce 100%)',
              color: 'var(--text-dark)',
              padding: '14px 32px', borderRadius: '999px',
              fontSize: '1rem', fontWeight: '600',
              boxShadow: '0 4px 20px rgba(201,192,227,0.45)',
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}
            onMouseEnter={e => { e.target.style.transform='translateY(-2px)'; e.target.style.boxShadow='0 6px 28px rgba(201,192,227,0.55)' }}
            onMouseLeave={e => { e.target.style.transform='translateY(0)'; e.target.style.boxShadow='0 4px 20px rgba(201,192,227,0.45)' }}
          >Analyze My Voice →</button>

          <button
            onClick={() => navigate('how-it-works')}
            style={{
              background: 'transparent',
              border: '1.5px solid var(--lavender)',
              color: 'var(--text-mid)',
              padding: '14px 32px', borderRadius: '999px',
              fontSize: '1rem', fontWeight: '500',
              transition: 'background 0.2s',
            }}
            onMouseEnter={e => e.target.style.background='var(--lav-light)'}
            onMouseLeave={e => e.target.style.background='transparent'}
          >How It Works</button>
        </div>

        {/* Disclaimer */}
        <p style={{
          marginTop: '2rem', fontSize: '0.75rem',
          color: 'var(--text-soft)', fontStyle: 'italic',
        }}>
          ⚠ Research prototype only — not a medical diagnostic tool.
        </p>
      </section>

      {/* Stats row */}
      <section style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '1rem', marginBottom: '4rem',
      }}>
        {stats.map((s, i) => (
          <div key={i} className="card" style={{
            background: s.color,
            border: `1.5px solid ${s.accent}`,
            textAlign: 'center', padding: '1.5rem 1rem',
            boxShadow: 'none',
          }}>
            <div style={{
              fontFamily: 'Cormorant Garamond, serif',
              fontSize: '2.4rem', fontWeight: 600,
              color: 'var(--text-dark)',
            }}>{s.value}</div>
            <div style={{ fontSize: '0.82rem', color: 'var(--text-mid)', marginTop: '4px' }}>{s.label}</div>
          </div>
        ))}
      </section>

      {/* Features grid */}
      <section>
        <h2 style={{
          fontFamily: 'Cormorant Garamond, serif',
          fontSize: '2.2rem', fontWeight: 300,
          textAlign: 'center', marginBottom: '0.5rem',
        }}>What Speak-ALS Does</h2>
        <p style={{ textAlign: 'center', color: 'var(--text-soft)', marginBottom: '2.5rem', fontSize: '0.95rem' }}>
          A full clinical-grade acoustic analysis pipeline in your browser
        </p>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
          gap: '1.2rem',
        }}>
          {features.map((f, i) => (
            <div key={i} className="card" style={{
              transition: 'transform 0.2s, box-shadow 0.2s',
            }}
              onMouseEnter={e => e.currentTarget.style.transform='translateY(-4px)'}
              onMouseLeave={e => e.currentTarget.style.transform='translateY(0)'}
            >
              <div style={{ fontSize: '2rem', marginBottom: '0.8rem' }}>{f.icon}</div>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 600, marginBottom: '0.5rem', fontFamily: 'Cormorant Garamond, serif' }}>{f.title}</h3>
              <p style={{ fontSize: '0.88rem', color: 'var(--text-mid)', lineHeight: 1.6 }}>{f.desc}</p>
            </div>
          ))}
        </div>
      </section>

      {/* CTA Banner */}
      <section style={{
        marginTop: '4rem',
        background: 'linear-gradient(135deg, #ede9f8 0%, #fae8ec 50%, #e0f4f0 100%)',
        borderRadius: 'var(--radius-lg)',
        padding: '3rem 2rem',
        textAlign: 'center',
        border: '1.5px solid rgba(201,192,227,0.4)',
      }}>
        <h2 style={{ fontSize: '2rem', fontWeight: 300, marginBottom: '1rem' }}>
          Ready to try it?
        </h2>
        <p style={{ color: 'var(--text-mid)', marginBottom: '1.5rem', fontSize: '0.95rem' }}>
          Record yourself saying "ahhhhh" for 3–5 seconds and upload the file.
        </p>
        <button
          onClick={() => navigate('analyze')}
          style={{
            background: 'linear-gradient(135deg, #c9c0e3, #f2c4ce)',
            color: 'var(--text-dark)',
            padding: '14px 36px', borderRadius: '999px',
            fontSize: '1rem', fontWeight: '600',
            boxShadow: '0 4px 20px rgba(201,192,227,0.4)',
          }}
        >Start Analysis →</button>
      </section>
    </div>
  )
}
