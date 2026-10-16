import { Link } from 'react-router-dom'
import { Code2 } from 'lucide-react'
import Logo from './Logo'
import { useApiStatus } from '../utils/apiStatus'

const REPO_URL = 'https://github.com/sumathic24/anti-deep'

function ApiStatus() {
  const { status } = useApiStatus()
  const { dot, label } = {
    checking: { dot: 'bg-fg-3', label: 'Checking API…' },
    online: { dot: 'bg-emerald-500', label: 'API online' },
    offline: { dot: 'bg-ai', label: 'API offline' },
  }[status]

  return (
    <span className="flex items-center gap-2 text-xs text-fg-3" role="status">
      <span className={`h-2 w-2 rounded-full ${dot}`} /> {label}
    </span>
  )
}

export default function Footer() {
  return (
    <footer className="mt-auto border-t border-line print:hidden">
      <div className="container-page py-10">
        <div className="grid grid-cols-2 gap-8 md:grid-cols-4">
          <div className="col-span-2">
            <Logo size="sm" />
            <p className="mt-3 max-w-sm text-sm leading-relaxed text-fg-3">
              Explainable detection of AI-generated text and deepfake images.
              Aligned with UN SDG 16: Peace, Justice and Strong Institutions.
            </p>
          </div>
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">Product</p>
            <ul className="space-y-2 text-sm">
              {[['/text-analysis', 'Text Analysis'], ['/image-analysis', 'Image Analysis'], ['/dashboard', 'Dashboard']].map(([path, label]) => (
                <li key={path}><Link to={path} className="text-fg-2 transition-colors hover:text-fg">{label}</Link></li>
              ))}
            </ul>
          </div>
          <div>
            <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">Project</p>
            <ul className="space-y-2 text-sm">
              <li><Link to="/about" className="text-fg-2 transition-colors hover:text-fg">About</Link></li>
              <li>
                <a href={REPO_URL} target="_blank" rel="noreferrer" className="inline-flex items-center gap-1.5 text-fg-2 transition-colors hover:text-fg">
                  <Code2 size={14} /> Source code
                </a>
              </li>
            </ul>
          </div>
        </div>
        <div className="mt-10 flex flex-col gap-3 border-t border-line pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="text-xs text-fg-3">
            © {new Date().getFullYear()} DeepFakeShield · Final Year Major Project, B.Tech CSE (Data Science)
          </p>
          <ApiStatus />
        </div>
      </div>
    </footer>
  )
}
