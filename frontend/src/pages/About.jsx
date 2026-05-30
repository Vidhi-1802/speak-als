const team = [
  {
    name: 'Kunj Rathi',
    role: 'Developer — AIML Engineer',
    desc: 'Final year engineering student specializing in Artificial Intelligence & Machine Learning. Built the full ML pipeline, feature extraction, and web application.',
    color: 'var(--lav-light)', accent: 'var(--lavender)', initial: 'K',
  },
]

const techStack = [
  { name: 'Python 3.10',     category: 'Core',        color: 'var(--lav-light)' },
  { name: 'scikit-learn',    category: 'ML',          color: 'var(--lav-light)' },
  { name: 'TensorFlow 2.13', category: 'Deep Learning', color: 'var(--blush-light)' },
  { name: 'Keras',           category: 'Deep Learning', color: 'var(--blush-light)' },
  { name: 'librosa',         category: 'Audio',       color: 'var(--mint-light)' },
  { name: 'parselmouth',     category: 'Audio',       color: 'var(--mint-light)' },
  { name: 'FastAPI',         category: 'Backend',     color: 'var(--peach-light)' },
  { name: 'React',           category: 'Frontend',    color: 'var(--peach-light)' },
  { name: 'numpy / pandas',  category: 'Data',        color: 'var(--lav-light)' },
  { name: 'matplotlib',      category: 'Viz',         color: 'var(--mint-light)' },
]

const refs = [
  'Little MA et al. (2008). Suitability of dysphonia measurements for telemonitoring of Parkinson\'s disease. IEEE TNSRE.',
  'Sakar BE et al. (2013). Collection and analysis of a Parkinson speech dataset with multiple types of sound recordings. IEEE JBHI.',
  'UCI Machine Learning Repository — Parkinson\'s Voice Dataset.',
  'Parselmouth — Jadoul Y. et al. (2018). Introducing Parselmouth: A Python interface to Praat. Journal of Phonetics.',
]

export default function About() {
  return (
    <div className="page-enter" style={{ maxWidth: 900, margin: '0 auto', padding: '3rem 2rem 5rem' }}>

      {/* Header */}
      <div style={{ textAlign: 'center', marginBottom: '3.5rem' }}>
        <div className="pill" style={{ background: 'var(--blush-light)', color: 'var(--text-mid)', marginBottom: '1rem' }}>
          About This Project
        </div>
        <h1 style={{ fontSize: '2.8rem', fontWeight: 300, marginBottom: '0.75rem' }}>Speak-ALS</h1>
        <p style={{ color: 'var(--text-mid)', maxWidth: 580, margin: '0 auto', lineHeight: 1.7, fontSize: '0.95rem' }}>
          A final year engineering project exploring the use of acoustic speech analysis and 
          machine learning for early detection of motor speech disorders associated with ALS.
        </p>
      </div>

      {/* Why this matters */}
      <div className="card" style={{ marginBottom: '2rem', background: 'linear-gradient(135deg, var(--lav-light), var(--blush-light))', boxShadow: 'none', border: '1.5px solid var(--lavender)' }}>
        <h2 style={{ fontSize: '1.6rem', fontWeight: 300, marginBottom: '1rem' }}>Why This Matters</h2>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-mid)', lineHeight: 1.8, marginBottom: '1rem' }}>
          Amyotrophic Lateral Sclerosis (ALS) is a progressive neurodegenerative disease that affects motor neurons.
          One of its earliest symptoms is <strong>dysarthria</strong> — a motor speech disorder that causes subtle but measurable 
          changes in voice quality, pitch stability, and articulation.
        </p>
        <p style={{ fontSize: '0.9rem', color: 'var(--text-mid)', lineHeight: 1.8 }}>
          Current ALS diagnosis relies on clinical visits and invasive testing. By detecting acoustic signatures 
          in voice recordings, Speak-ALS explores the possibility of a <strong>non-invasive, accessible pre-screening tool</strong> 
          that could help identify at-risk individuals earlier, enabling faster medical evaluation.
        </p>
      </div>

      {/* Model performance */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 300, marginBottom: '1.5rem' }}>Model Performance</h2>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '0.88rem' }}>
            <thead>
              <tr style={{ borderBottom: '2px solid rgba(201,192,227,0.3)' }}>
                {['Metric', 'SVM', 'CNN', 'Winner'].map(h => (
                  <th key={h} style={{ padding: '10px 12px', textAlign: 'left', color: 'var(--text-soft)', fontWeight: 600, fontSize: '0.78rem', textTransform: 'uppercase', letterSpacing: '0.06em' }}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {[
                ['Accuracy',  '89.7%', '92.3%', 'CNN ✓'],
                ['Precision', '93.1%', '96.4%', 'CNN ✓'],
                ['Recall',    '93.1%', '93.1%', 'Tie'],
                ['F1-Score',  '93.1%', '94.7%', 'CNN ✓'],
                ['ROC-AUC',   '96.2%', '97.6%', 'CNN ✓'],
              ].map((row, i) => (
                <tr key={i} style={{ borderBottom: '1px solid rgba(201,192,227,0.15)', background: i % 2 === 0 ? 'transparent' : 'rgba(201,192,227,0.04)' }}>
                  {row.map((cell, j) => (
                    <td key={j} style={{
                      padding: '10px 12px', color: j === 3 ? '#a08cc0' : 'var(--text-dark)',
                      fontWeight: j === 3 ? 600 : 400,
                    }}>{cell}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Team */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 300, marginBottom: '1.5rem' }}>Built By</h2>
        {team.map((m, i) => (
          <div key={i} style={{ display: 'flex', gap: '1.2rem', alignItems: 'flex-start' }}>
            <div style={{
              width: 52, height: 52, borderRadius: '50%', flexShrink: 0,
              background: `linear-gradient(135deg, ${m.accent}, var(--blush))`,
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              fontFamily: 'Cormorant Garamond, serif', fontSize: '1.4rem', fontWeight: 600,
            }}>{m.initial}</div>
            <div>
              <div style={{ fontWeight: 600, fontSize: '1.05rem' }}>{m.name}</div>
              <div style={{ fontSize: '0.82rem', color: '#a08cc0', marginBottom: '6px' }}>{m.role}</div>
              <p style={{ fontSize: '0.85rem', color: 'var(--text-mid)', lineHeight: 1.6 }}>{m.desc}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Tech stack */}
      <div className="card" style={{ marginBottom: '2rem' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 300, marginBottom: '1.2rem' }}>Technology Stack</h2>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.6rem' }}>
          {techStack.map((t, i) => (
            <div key={i} style={{
              background: t.color, padding: '6px 14px', borderRadius: 999,
              fontSize: '0.82rem', color: 'var(--text-dark)',
            }}>
              <span style={{ fontWeight: 600 }}>{t.name}</span>
              <span style={{ color: 'var(--text-soft)', marginLeft: 4 }}>· {t.category}</span>
            </div>
          ))}
        </div>
      </div>

      {/* References */}
      <div className="card" style={{ background: 'var(--mint-light)', boxShadow: 'none', border: '1.5px solid var(--mint)' }}>
        <h2 style={{ fontSize: '1.5rem', fontWeight: 300, marginBottom: '1rem' }}>References</h2>
        <ol style={{ paddingLeft: '1.2rem', display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
          {refs.map((r, i) => (
            <li key={i} style={{ fontSize: '0.82rem', color: 'var(--text-mid)', lineHeight: 1.6 }}>{r}</li>
          ))}
        </ol>
      </div>
    </div>
  )
}
