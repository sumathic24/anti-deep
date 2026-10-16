import { useState, useEffect } from 'react'
import { Link, NavLink, useLocation } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { Menu, X, ScanSearch } from 'lucide-react'
import Logo from './Logo'

const navLinks = [
  { label: 'Home', path: '/' },
  { label: 'Text Analysis', path: '/text-analysis' },
  { label: 'Image Analysis', path: '/image-analysis' },
  { label: 'Dashboard', path: '/dashboard' },
  { label: 'About', path: '/about' },
]

export default function Navbar() {
  const location = useLocation()
  const [menuOpen, setMenuOpen] = useState(false)

  useEffect(() => setMenuOpen(false), [location])

  const linkClass = ({ isActive }) =>
    `relative rounded-md px-3 py-2 text-sm font-medium transition-colors ${isActive ? 'text-fg' : 'text-fg-2 hover:text-fg'}`

  return (
    <header className="sticky top-0 z-50 border-b border-line bg-canvas/85 backdrop-blur-md">
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

        <div className="hidden md:block">
          <Link to="/text-analysis" className="btn-primary btn-sm">
            <ScanSearch size={14} /> Analyze
          </Link>
        </div>

        <button type="button" className="btn-ghost -mr-2 p-2 md:hidden" onClick={() => setMenuOpen(!menuOpen)}
          aria-label={menuOpen ? 'Close menu' : 'Open menu'} aria-expanded={menuOpen}>
          {menuOpen ? <X size={20} /> : <Menu size={20} />}
        </button>
      </nav>

      <AnimatePresence>
        {menuOpen && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }}
            className="overflow-hidden border-t border-line md:hidden">
            <div className="container-page flex flex-col gap-1 py-3">
              {navLinks.map((link) => (
                <NavLink key={link.path} to={link.path} end={link.path === '/'}
                  className={({ isActive }) =>
                    `rounded-lg px-3 py-2.5 text-sm font-medium ${isActive ? 'bg-white/5 text-fg' : 'text-fg-2 hover:bg-white/5 hover:text-fg'}`}>
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
