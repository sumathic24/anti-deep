import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { AnimatePresence } from 'framer-motion'
import Navbar from './components/Navbar'
import Landing from './pages/Landing'
import Dashboard from './pages/Dashboard'
import TextAnalysis from './pages/TextAnalysis'
import ImageAnalysis from './pages/ImageAnalysis'
import About from './pages/About'
import Footer from './components/Footer'

export default function App() {
  return (
    <Router>
      <div className="min-h-screen bg-dark-950 relative overflow-x-hidden">
        {/* Global background orbs */}
        <div className="fixed inset-0 pointer-events-none overflow-hidden z-0">
          <div className="orb-cyan" style={{ top: '-200px', left: '-100px' }} />
          <div className="orb-violet" style={{ top: '200px', right: '-200px' }} />
          <div className="orb-cyan" style={{ bottom: '-100px', left: '40%' }} />
        </div>

        <Navbar />

        <AnimatePresence mode="wait">
          <Routes>
            <Route path="/" element={<Landing />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/text-analysis" element={<TextAnalysis />} />
            <Route path="/image-analysis" element={<ImageAnalysis />} />
            <Route path="/about" element={<About />} />
          </Routes>
        </AnimatePresence>

        <Footer />
      </div>
    </Router>
  )
}
