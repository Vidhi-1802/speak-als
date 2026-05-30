import { useState, useRef } from 'react'

const API = 'http://localhost:8000'

const tips = [
  'Record in a quiet room with minimal background noise',
  'Sustain the "ahhh" sound for at least 3 seconds',
  'Hold the microphone about 15–20 cm from your mouth',
  'Save as .wav format before uploading',
]

export default function Analyze({ onResults }) {
  const [file, setFile]         = useState(null)
  const [dragging, setDragging] = useState(false)
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const [step, setStep]         = useState(0) // 0=idle, 1=uploading, 2=extracting, 3=predicting
  const fileRef = useRef()

  const steps = ['Uploading audio…', 'Extracting acoustic features…', 'Running SVM + CNN models…']

  const handleFile = (f) => {
    if (!f) return
    if (!f.name.endsWith('.wav') && !f.name.endsWith('.WAV')) {
      setError('Please upload a .wav file only.')
      return
    }
    setFile(f)
    setError(null)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true)
    setError(null)

    try {
      // Simulate step progression visually
      setStep(1)
      await new Promise(r => setTimeout(r, 600))
      setStep(2)
      await new Promise(r => setTimeout(r, 800))
      setStep(3)

      const formData = new FormData()
      formData.append('file', file)

      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        body: formData,
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Analysis failed')
      }

      const data = await res.json()
      onResults(data)
    } catch (e) {
      setError(e.message.includes('fetch') 
        ? 'Cannot connect to the backend. Make sure the FastAPI server is running on port 8000.'
        : e.message)
      setLoading(false)
      setStep(0)
    }
  }

  return (
    <div className="page-enter" style={{ maxWidth: 780, margin: '0 auto', padding: '3rem 2rem 5rem' }}>
      <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
        <div className="pill" style={{ background: 'var(--lav-light)', color: 'var(--text-mid)', marginBottom: '1rem' }}>
          Step 1 of 1 — Upload
        </div>
        <h1 style={{ fontSize: '2.8rem', fontWeight: 300, marginBottom: '0.75rem' }}>
          Upload Your Voice Recording
        </h1>
        <p style={{ color: 'var(--text-mid)', fontSize: '0.95rem' }}>
          Record yourself saying <strong>"ahhhhh"</strong> for 3–5 seconds and upload the .wav file
        </p>
      </div>

      {/* Upload zone */}
      <div
        onClick={() => !loading && fileRef.current.click()}
        onDragOver={e => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={handleDrop}
        style={{
          border: `2px dashed ${dragging ? '#c9c0e3' : file ? '#b8ddd4' : 'rgba(201,192,227,0.5)'}`,
          borderRadius: 'var(--radius-lg)',
          padding: '3rem 2rem',
          textAlign: 'center',
          cursor: loading ? 'not-allowed' : 'pointer',
          background: dragging ? 'var(--lav-light)' : file ? 'var(--mint-light)' : 'var(--white)',
          transition: 'all 0.25s',
          marginBottom: '1.5rem',
          boxShadow: 'var(--shadow-card)',
        }}
      >
        <input
          ref={fileRef}
          type="file"
          accept=".wav"
          style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />

        {file ? (
          <>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>✅</div>
            <p style={{ fontWeight: 600, color: 'var(--text-dark)', fontSize: '1.05rem' }}>{file.name}</p>
            <p style={{ color: 'var(--text-soft)', fontSize: '0.85rem', marginTop: '4px' }}>
              {(file.size / 1024).toFixed(1)} KB · Click to change file
            </p>
          </>
        ) : (
          <>
            <div style={{ fontSize: '2.5rem', marginBottom: '0.75rem' }}>🎙</div>
            <p style={{ fontWeight: 500, color: 'var(--text-dark)' }}>Drop your .wav file here</p>
            <p style={{ color: 'var(--text-soft)', fontSize: '0.85rem', marginTop: '6px' }}>or click to browse</p>
          </>
        )}
      </div>

      {/* Tips */}
      <div className="card" style={{ marginBottom: '2rem', background: 'var(--peach-light)', boxShadow: 'none', border: '1.5px solid var(--peach)' }}>
        <p style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-mid)', marginBottom: '0.75rem', letterSpacing: '0.06em', textTransform: 'uppercase' }}>Recording Tips</p>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
          {tips.map((t, i) => (
            <div key={i} style={{ display: 'flex', gap: '8px', alignItems: 'flex-start' }}>
              <span style={{ color: 'var(--text-soft)', marginTop: 2 }}>•</span>
              <p style={{ fontSize: '0.84rem', color: 'var(--text-mid)', lineHeight: 1.5 }}>{t}</p>
            </div>
          ))}
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          background: '#fff0f3', border: '1.5px solid #f2c4ce',
          borderRadius: 'var(--radius-md)', padding: '1rem 1.2rem',
          marginBottom: '1.5rem', fontSize: '0.88rem', color: '#a04060',
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Loading steps */}
      {loading && (
        <div className="card" style={{ marginBottom: '1.5rem', textAlign: 'center' }}>
          <div style={{ marginBottom: '1rem' }}>
            {steps.map((s, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: '10px',
                padding: '8px 0', opacity: step >= i + 1 ? 1 : 0.3,
                transition: 'opacity 0.4s',
              }}>
                <div style={{
                  width: 20, height: 20, borderRadius: '50%',
                  background: step > i + 1 ? 'var(--mint)' : step === i + 1 ? 'var(--lavender)' : 'var(--lav-light)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center',
                  fontSize: '0.7rem', flexShrink: 0,
                  animation: step === i + 1 ? 'pulse 1s infinite' : 'none',
                }}>
                  {step > i + 1 ? '✓' : i + 1}
                </div>
                <span style={{ fontSize: '0.9rem', color: 'var(--text-mid)' }}>{s}</span>
              </div>
            ))}
          </div>
          <style>{`@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }`}</style>
        </div>
      )}

      {/* Analyze button */}
      <button
        onClick={handleAnalyze}
        disabled={!file || loading}
        style={{
          width: '100%',
          background: file && !loading
            ? 'linear-gradient(135deg, #c9c0e3 0%, #f2c4ce 100%)'
            : 'rgba(201,192,227,0.3)',
          color: file && !loading ? 'var(--text-dark)' : 'var(--text-soft)',
          padding: '16px', borderRadius: 'var(--radius-md)',
          fontSize: '1.05rem', fontWeight: '600',
          boxShadow: file && !loading ? '0 4px 20px rgba(201,192,227,0.4)' : 'none',
          transition: 'all 0.2s',
          cursor: file && !loading ? 'pointer' : 'not-allowed',
        }}
      >
        {loading ? 'Analyzing…' : 'Analyze Speech →'}
      </button>
    </div>
  )
}
