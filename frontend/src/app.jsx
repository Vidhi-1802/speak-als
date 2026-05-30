import { useState } from 'react'
import Navbar from './components/Navbar.jsx'
import Home from './pages/Home.jsx'
import Analyze from './pages/Analyze.jsx'
import Results from './pages/Results.jsx'
import HowItWorks from './pages/HowItWorks.jsx'
import About from './pages/About.jsx'
import './index.css'

export default function App() {
  const [page, setPage] = useState('home')
  const [results, setResults] = useState(null)

  const navigate = (p) => {
    setPage(p)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  const handleResults = (data) => {
    setResults(data)
    navigate('results')
  }

  return (
    <div style={{ position: 'relative', zIndex: 1 }}>
      <Navbar currentPage={page} navigate={navigate} />
      <main>
        {page === 'home'        && <Home navigate={navigate} />}
        {page === 'analyze'     && <Analyze onResults={handleResults} />}
        {page === 'results'     && <Results data={results} navigate={navigate} />}
        {page === 'how-it-works'&& <HowItWorks />}
        {page === 'about'       && <About />}
      </main>
    </div>
  )
}
