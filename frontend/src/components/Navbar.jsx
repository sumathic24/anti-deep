import { useState, useEffect, useRef } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, Sun, Moon } from 'lucide-react'
import Logo from './Logo'
import { useTheme } from '../utils/theme'

const navLinks = [
  { label: 'Home', path: '/' },
  { label: 'Text Analysis', path: '/text-analysis' },
  { label: 'Image Analysis', path: '/image-analysis' },
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'About', path: '/about' },
]

function ThemeToggle() {
  const { theme, toggle } = useTheme()
  const next = theme === 'dark' ? 'light' : 'dark'
  return (
    <button type="button" onClick={toggle} className="btn-ghost p-2" aria-label={`Switch to ${next} theme`} title={`Switch to ${next} theme`}>
      {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
    </button>
  )
}

export default function Navbar() {
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)
  const headerRef = useRef(null)

  useEffect(() => setMenuOpen(false), [location])

  // Close the mobile menu with Escape or a tap outside it.
  useEffect(() => {
    if (!menuOpen) return
    const onKey = (e) => e.key === 'Escape' && setMenuOpen(false)
    const onPointer = (e) => headerRef.current && !headerRef.current.contains(e.target) && setMenuOpen(false)
    document.addEventListener('keydown', onKey)
    document.addEventListener('pointerdown', onPointer)
    return () => {
      document.removeEventListener('keydown', onKey)
      document.removeEventListener('pointerdown', onPointer)
    }
  }, [menuOpen])

  const linkClass = ({ isActive }) =>
    `relative rounded-md px-3 py-2 text-sm font-medium transition-colors ${isActive ? 'text-fg' : 'text-fg-2 hover:text-fg'}`

  return (
    <header ref={headerRef} className="sticky top-0 z-50 border-b border-line bg-canvas/85 backdrop-blur-md print:hidden">
      <nav className="container-page flex h-16 items-center justify-between" aria-label="Main">
        <Link to="/" className="flex items-center gap-2.5" aria-label="DeepFakeShield home">
          <Logo />
        </Link>

        <div className="hidden items-center gap-1 md:flex">
          {navLinks.map((link) => (
            <NavLink key={link.path} to={link.path} end={link.path === '/'} className={linkClass}>
              {({ isActive }) => (
                <>
                  {link.label}
                  {isActive && (
                    <motion.span layoutId="nav-underline" className="absolute inset-x-3 -bottom-[13px] h-0.5 rounded-full bg-accent"
                      transition={{ type: 'spring', stiffness: 400, damping: 35 }} />
                  )}
                </>
              )}
            </NavLink>
          ))}
        </div>

        <div className="flex items-center gap-1">
          <ThemeToggle />
          <button type="button" className="btn-ghost -mr-2 p-2 md:hidden" onClick={() => setMenuOpen(!menuOpen)}
            aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen} aria-controls="mobile-menu">
            {menuOpen ? <X size={20} /> : <Menu size={20} />}
          </button>
        </div>
      </nav>

      <AnimatePresence>
        {menuOpen && (
          <motion.div id="mobile-menu" initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-line md:hidden">
            <div className="container-page flex flex-col gap-1 py-3">
              {navLinks.map((link) => (
                <NavLink key={link.path} to={link.path} end={link.path === '/'}
                  className={({ isActive }) =>
                    `rounded-lg px-3 py-2.5 text-sm font-medium ${isActive ? 'bg-fg/5 text-fg' : 'text-fg-2 hover:bg-fg/5 hover:text-fg'}`}>
                  {link.label}
                </NavLink>
              ))}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </header>
  )
}
