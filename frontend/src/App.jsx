import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { MotionConfig } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Landing from './pages/Landing'

// Pages other than the landing page load on demand to keep the first download small.
const Dashboard = lazy(() => import('./pages/Dashboard'))
const TextAnalysis = lazy(() => import('./pages/TextAnalysis'))
const ImageAnalysis = lazy(() => import('./pages/ImageAnalysis'))
const About = lazy(() => import('./pages/About'))
const NotFound = lazy(() => import('./pages/NotFound'))

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => { window.scrollTo(0, 0) }, [pathname])
  return null
}

function PageFallback() {
  return (
    <main className="flex flex-1 items-center justify-center py-32">
      <Loader2 size={24} className="animate-spin text-fg-3" aria-label="Loading page" />
    </main>
  )
}

export default function App() {
  return (
    <Router>
      <MotionConfig reducedMotion="user">
        <ScrollToTop />
        <div className="flex min-h-screen flex-col">
          <Navbar />
          <Suspense fallback={<PageFallback />}>
            <Routes>
              <Route path="/" element={<Landing />} />
              <Route path="/dashboard" element={<Dashboard />} />
              <Route path="/text-analysis" element={<TextAnalysis />} />
              <Route path="/image-analysis" element={<ImageAnalysis />} />
              <Route path="/about" element={<About />} />
              <Route path="*" element={<NotFound />} />
            </Routes>
          </Suspense>
          <Footer />
        </div>
      </MotionConfig>
    </Router>
  )
}
