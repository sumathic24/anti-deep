import { lazy, Suspense, useEffect } from 'react'
import { BrowserRouter as Router, Routes, Route, useLocation } from 'react-router-dom'
import { MotionConfig } from 'framer-motion'
import { Loader2 } from 'lucide-react'
import Navbar from './components/Navbar'
import Footer from './components/Footer'
import Landing from './pages/Landing'
import { ThemeProvider } from './utils/theme'
import { ApiStatusProvider } from './utils/apiStatus'

// Pages other than the landing page load on demand to keep the first download small.
const Dashboard = lazy(() => import('./pages/Dashboard'))
const TextAnalysis = lazy(() => import('./pages/TextAnalysis'))
const ImageAnalysis = lazy(() => import('./pages/ImageAnalysis'))
const About = lazy(() => import('./pages/About'))
const NotFound = lazy(() => import('./pages/NotFound'))

/** Scroll to the top on navigation, or to #section when the link has a hash. */
function ScrollManager() {
  const { pathname, hash } = useLocation()
  useEffect(() => {
    if (!hash) { window.scrollTo(0, 0); return }
    // Lazy pages render a moment later, so retry briefly until the target exists.
    let tries = 0
    const timer = setInterval(() => {
      const el = document.getElementById(hash.slice(1))
      if (el || ++tries > 20) {
        clearInterval(timer)
        el?.scrollIntoView({ behavior: 'smooth', block: 'start' })
      }
    }, 100)
    return () => clearInterval(timer)
  }, [pathname, hash])
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
    <ThemeProvider>
      <ApiStatusProvider>
        <Router>
          <MotionConfig reducedMotion="user">
            <ScrollManager />
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
      </ApiStatusProvider>
    </ThemeProvider>
  )
}
