import { useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, AlertTriangle, CheckCircle2, Info, Loader2, Copy, Check, Printer, WifiOff } from 'lucide-react'
import { useTheme } from '../utils/theme'
import { useApiStatus } from '../utils/apiStatus'

export const fadeIn = {
  hidden: { opacity: 0, y: 8 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.35, delay: i * 0.05, ease: 'easeOut' } }),
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show"
      className="mb-8 flex flex-col gap-4 print:hidden sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">{title}</h1>
        {description && <p className="mt-2 text-sm leading-relaxed text-fg-2 sm:text-base">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap gap-2">{actions}</div>}
    </motion.div>
  )
}

export function Alert({ tone = 'error', icon, children }) {
  const styles = tone === 'error' ? 'border-ai/30 bg-ai/10 text-ai-text' : 'border-line-strong bg-raised text-fg-2'
  const Icon = icon || (tone === 'error' ? AlertTriangle : Info)
  return (
    <div role={tone === 'error' ? 'alert' : 'note'} className={`flex items-start gap-2.5 rounded-lg border px-3.5 py-3 text-sm ${styles}`}>
      <Icon size={16} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1">{children}</div>
    </div>
  )
}

/** Warns before the user starts typing when the backend can't be reached. */
export function ApiOfflineBanner() {
  const { status, recheck } = useApiStatus()
  if (status !== 'offline') return null
  return (
    <div className="mb-6 print:hidden">
      <Alert icon={WifiOff}>
        <span className="font-medium">The analysis server isn't reachable.</span>{' '}
        Start the backend (see README), then{' '}
        <button type="button" onClick={recheck} className="font-medium underline underline-offset-2">check again</button>.
      </Alert>
    </div>
  )
}

export function VerdictBadge({ isAI, children }) {
  const { colors } = useTheme()
  const mark = isAI ? colors.ai : colors.human
  const Icon = isAI ? AlertTriangle : CheckCircle2
  return (
    <span className="badge" style={{ color: isAI ? colors.aiText : colors.humanText, background: `${mark}1f`, borderColor: `${mark}59` }}>
      <Icon size={13} /> {children}
    </span>
  )
}

/** One horizontal bar split into the AI share and the human/real share. */
export function SplitBar({ ai, aiLabel = 'AI-generated', otherLabel = 'Human-written' }) {
  const { colors } = useTheme()
  const aiPct = ai * 100
  return (
    <div>
      <div className="mb-2 flex justify-between gap-3 text-xs">
        <span className="flex items-center gap-1.5 text-fg-2">
          <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: colors.ai }} />
          {aiLabel} <span className="tabular font-medium text-fg">{aiPct.toFixed(1)}%</span>
        </span>
        <span className="flex items-center gap-1.5 text-right text-fg-2">
          <span className="tabular font-medium text-fg">{(100 - aiPct).toFixed(1)}%</span> {otherLabel}
          <span className="h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: colors.human }} />
        </span>
      </div>
      <div className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-full">
        <motion.div initial={{ width: 0 }} animate={{ width: `${aiPct}%` }} transition={{ duration: 0.6, ease: 'easeOut' }}
          className="h-full" style={{ background: colors.ai }} />
        <div className="h-full flex-1" style={{ background: colors.human }} />
      </div>
    </div>
  )
}

/** Headline result: verdict, confidence in that verdict, the AI/other split and run metadata. */
export function VerdictCard({ result, otherLabel, actions }) {
  const isAI = result.ai_confidence > 0.5
  const confidence = Math.max(result.ai_confidence, 1 - result.ai_confidence) * 100
  return (
    <div className="card p-5 sm:p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-fg-3">Verdict</p>
          <VerdictBadge isAI={isAI}>{result.prediction}</VerdictBadge>
        </div>
        <div className="text-right">
          <p className="text-4xl font-semibold tracking-tight text-fg">{confidence.toFixed(1)}%</p>
          <p className="text-xs text-fg-3">confidence in this verdict</p>
        </div>
      </div>
      <SplitBar ai={result.ai_confidence} otherLabel={otherLabel} />
      <dl className="mt-5 grid grid-cols-1 gap-3 border-t border-line pt-4 text-xs sm:grid-cols-[1fr_auto]">
        <div className="min-w-0">
          <dt className="text-fg-3">Model</dt>
          <dd className="mt-0.5 break-words text-fg-2">{result.model_used}</dd>
        </div>
        <div className="sm:text-right">
          <dt className="text-fg-3">Processing time</dt>
          <dd className="tabular mt-0.5 text-fg-2">{(result.processing_time_ms / 1000).toFixed(result.processing_time_ms < 1000 ? 2 : 1)} s</dd>
        </div>
      </dl>
      {actions && <div className="mt-4 flex flex-wrap gap-2 border-t border-line pt-4 print:hidden">{actions}</div>}
    </div>
  )
}

/** Buttons shared by the analysis pages: copy a plain-text summary and print / save a PDF report. */
export function ReportActions({ summary }) {
  const [copyState, setCopyState] = useState('idle') // idle | copied | failed
  const copy = async () => {
    let ok = false
    try {
      await navigator.clipboard.writeText(summary)
      ok = true
    } catch {
      // The Clipboard API is missing on plain-http addresses (e.g. a LAN IP); fall back to execCommand.
      const ta = document.createElement('textarea')
      ta.value = summary
      ta.setAttribute('readonly', '')
      ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none'
      document.body.appendChild(ta)
      ta.select()
      try { ok = document.execCommand('copy') } catch { ok = false }
      ta.remove()
    }
    setCopyState(ok ? 'copied' : 'failed')
    setTimeout(() => setCopyState('idle'), 2500)
  }
  return (
    <>
      <button type="button" className="btn-secondary btn-sm" onClick={copy}>
        {copyState === 'copied' ? <><Check size={14} /> Copied</>
          : copyState === 'failed' ? <><Copy size={14} /> Copy failed — select the text manually</>
          : <><Copy size={14} /> Copy summary</>}
      </button>
      <button type="button" className="btn-secondary btn-sm" onClick={() => window.print()}>
        <Printer size={14} /> Print / save as PDF
      </button>
    </>
  )
}

export function CollapsibleCard({ title, icon: Icon, description, defaultOpen = true, children }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div className="card">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="flex w-full items-center justify-between gap-3 rounded-xl px-5 py-4 text-left sm:px-6">
        <span className="flex items-center gap-2.5">
          {Icon && <Icon size={16} className="text-fg-3" />}
          <span className="text-sm font-semibold text-fg">{title}</span>
        </span>
        <ChevronDown size={16} className={`text-fg-3 transition-transform ${open ? 'rotate-180' : ''}`} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }} transition={{ duration: 0.2 }} className="overflow-hidden">
            <div className="border-t border-line px-5 pb-5 pt-4 sm:px-6">
              {description && <p className="mb-4 text-xs leading-relaxed text-fg-3">{description}</p>}
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

export function EmptyState({ icon: Icon, title, description, children }) {
  return (
    <div className="card flex flex-col items-center justify-center px-6 py-16 text-center">
      <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-line bg-raised">
        <Icon size={22} className="text-fg-3" />
      </div>
      <p className="font-medium text-fg">{title}</p>
      {description && <p className="mt-1 max-w-sm text-sm text-fg-3">{description}</p>}
      {children && <div className="mt-6">{children}</div>}
    </div>
  )
}

/** Spinner with elapsed time, optional upload progress and a Cancel button. */
export function LoadingState({ title, description, progress, onCancel }) {
  const [seconds, setSeconds] = useState(0)
  useEffect(() => {
    const started = Date.now()
    const timer = setInterval(() => setSeconds(Math.floor((Date.now() - started) / 1000)), 1000)
    return () => clearInterval(timer)
  }, [])
  return (
    <div role="status" className="card flex flex-col items-center justify-center px-6 py-16 text-center">
      <Loader2 size={28} className="mb-4 animate-spin text-accent" />
      <p className="font-medium text-fg">{title}</p>
      {description && <p className="mt-1 max-w-sm text-sm text-fg-3">{description}</p>}
      {progress != null && progress < 1 && (
        <div className="mt-5 w-56">
          <div className="meter h-1.5"><div className="h-full rounded-full bg-accent" style={{ width: `${progress * 100}%` }} /></div>
          <p className="tabular mt-1.5 text-xs text-fg-3">Uploading… {Math.round(progress * 100)}%</p>
        </div>
      )}
      <p className="tabular mt-4 text-xs text-fg-3">
        {seconds}s elapsed{seconds >= 10 && ' · large models on a CPU can take up to a minute'}
      </p>
      {onCancel && <button type="button" className="btn-secondary btn-sm mt-5" onClick={onCancel}>Cancel</button>}
    </div>
  )
}

/** Two-model comparison used by both analysis pages. */
export function ModelComparison({ models }) {
  const { colors } = useTheme()
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
      {models.map(({ name, ai, prediction, weight }) => (
        <div key={name} className="rounded-lg border border-line bg-raised p-4">
          <div className="mb-3 flex items-center justify-between">
            <p className="text-sm font-medium text-fg">{name}</p>
            {weight != null && <span className="tabular text-xs text-fg-3">weight {weight}</span>}
          </div>
          <p className="text-2xl font-semibold text-fg">{(ai * 100).toFixed(1)}%</p>
          <p className="mb-3 text-xs text-fg-3">AI likelihood</p>
          <div className="meter">
            <div className="h-full rounded-full" style={{ width: `${ai * 100}%`, background: colors.ai }} />
          </div>
          <p className="mt-2 text-xs text-fg-2">{prediction}</p>
        </div>
      ))}
    </div>
  )
}

/** Print-only report header shared by the analysis pages. */
export function ReportHeader({ title }) {
  return (
    <div className="mb-6 border-b border-line pb-4">
      <p className="text-sm font-semibold">DeepFakeShield</p>
      <h1 className="mt-1 text-2xl font-semibold">{title}</h1>
      <p className="mt-1 text-xs text-fg-3">Generated {new Date().toLocaleString()}</p>
    </div>
  )
}

/** Scrolls the results column into view on small screens, where it sits below the input. */
export function scrollToOnMobile(ref) {
  if (ref.current && window.matchMedia('(max-width: 1023px)').matches) {
    ref.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }
}
