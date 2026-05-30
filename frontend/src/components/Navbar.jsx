import { useState } from 'react'

const links = [
  { id: 'home',          label: 'Home' },
  { id: 'analyze',       label: 'Analyze' },
  { id: 'how-it-works',  label: 'How It Works' },
  { id: 'about',         label: 'About' },
]

export default function Navbar({ currentPage, navigate }) {
  const [menuOpen, setMenuOpen] = useState(false)

  return (
    <nav style={{
      position: 'sticky', top: 0, zIndex: 100,
      background: 'rgba(250,247,242,0.88)',
      backdropFilter: 'blur(16px)',
      borderBottom: '1px solid rgba(201,192,227,0.25)',
      padding: '0 2rem',
      display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      height: '64px',
    }}>
      {/* Logo */}
      <div
        onClick={() => navigate('home')}
        style={{ cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '10px' }}
      >
        <div style={{
          width: 36, height: 36, borderRadius: '50%',
          background: 'linear-gradient(135deg, #f2c4ce, #c9c0e3)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontSize: '1rem',
        }}>🎙</div>
        <span style={{
          fontFamily: 'Cormorant Garamond, serif',
          fontSize: '1.4rem', fontWeight: 600,
          color: 'var(--text-dark)',
        }}>Speak<span style={{ color: '#a08cc0' }}>ALS</span></span>
      </div>

      {/* Desktop links */}
      <div style={{ display: 'flex', gap: '2rem' }}>
        {links.map(l => (
          <button
            key={l.id}
            onClick={() => navigate(l.id)}
            style={{
              background: 'none',
              fontSize: '0.9rem',
              fontWeight: currentPage === l.id ? '600' : '400',
              color: currentPage === l.id ? 'var(--text-dark)' : 'var(--text-mid)',
              borderBottom: currentPage === l.id ? '2px solid #c9c0e3' : '2px solid transparent',
              paddingBottom: '2px',
              transition: 'all 0.2s',
            }}
          >{l.label}</button>
        ))}
      </div>

      {/* CTA */}
      <button
        onClick={() => navigate('analyze')}
        style={{
          background: 'linear-gradient(135deg, #c9c0e3, #f2c4ce)',
          color: 'var(--text-dark)',
          padding: '8px 20px',
          borderRadius: '999px',
          fontSize: '0.88rem',
          fontWeight: '600',
          boxShadow: '0 2px 12px rgba(201,192,227,0.4)',
          transition: 'transform 0.15s, box-shadow 0.15s',
        }}
        onMouseEnter={e => e.target.style.transform = 'translateY(-1px)'}
        onMouseLeave={e => e.target.style.transform = 'translateY(0)'}
      >
        Try Now →
      </button>
    </nav>
  )
}
