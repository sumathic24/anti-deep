import { createContext, useCallback, useContext, useEffect, useState } from 'react'

// Hex values for places that need real colors (charts, inline styles). Each pair of
// verdict colors is validated as colour-blind safe on that theme's surface.
export const PALETTE = {
  light: {
    ai: '#e34948', aiText: '#b42318', human: '#2a78d6', humanText: '#1d4ed8',
    surface: '#ffffff', grid: '#e3e6eb', axis: '#cfd5dd', tick: '#5d6878', fg: '#0f1726', fg2: '#3d4757',
    meta: '#f6f7f9',
  },
  dark: {
    ai: '#e66767', aiText: '#fca5a5', human: '#3987e5', humanText: '#93c5fd',
    surface: '#0d1320', grid: '#1c2536', axis: '#2a3549', tick: '#7c869a', fg: '#e6eaf2', fg2: '#a3acbf',
    meta: '#070b14',
  },
}

const ThemeContext = createContext({ theme: 'dark', toggle: () => {}, colors: PALETTE.dark })

export function ThemeProvider({ children }) {
  const [theme, setTheme] = useState(() =>
    document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark')

  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme)
    document.querySelector('meta[name="theme-color"]')?.setAttribute('content', PALETTE[theme].meta)
    try { localStorage.setItem('theme', theme) } catch { /* storage unavailable: theme still applies */ }
  }, [theme])

  const toggle = useCallback(() => setTheme((t) => (t === 'dark' ? 'light' : 'dark')), [])

  return (
    <ThemeContext.Provider value={{ theme, toggle, colors: PALETTE[theme] }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => useContext(ThemeContext)
