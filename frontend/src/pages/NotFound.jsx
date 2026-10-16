import { Link, useLocation } from 'react-router-dom'
import { Home, ScanSearch } from 'lucide-react'

export default function NotFound() {
  const { pathname } = useLocation()
  return (
    <main className="container-page flex flex-1 flex-col items-center justify-center py-24 text-center">
      <p className="eyebrow mb-3">Error 404</p>
      <h1 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">Page not found</h1>
      <p className="mt-3 max-w-md text-fg-2">
        There is no page at <code className="rounded bg-raised px-1.5 py-0.5 font-mono text-sm text-fg">{pathname}</code>.
        Check the address or go back to a known page.
      </p>
      <div className="mt-8 flex flex-wrap justify-center gap-3">
        <Link to="/" className="btn-primary"><Home size={16} /> Go home</Link>
        <Link to="/text-analysis" className="btn-secondary"><ScanSearch size={16} /> Analyze text</Link>
      </div>
    </main>
  )
}
