import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { ChevronDown, AlertTriangle, CheckCircle2, Info, Loader2 } from 'lucide-react'

// Verdict colors (marks) and their legible text tints — see tailwind.config.js
export const VERDICT_COLORS = {
  ai: { mark: '#e66767', text: '#fca5a5' },
  human: { mark: '#3987e5', text: '#93c5fd' },
}

export const fadeIn = {
  hidden: { opacity: 0, y: 8 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.35, delay: i * 0.05, ease: 'easeOut' } }),
}

export function PageHeader({ eyebrow, title, description, actions }) {
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show"
      className="mb-8 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div className="max-w-2xl">
        {eyebrow && <p className="eyebrow mb-2">{eyebrow}</p>}
        <h1 className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">{title}</h1>
        {description && <p className="mt-2 text-sm leading-relaxed text-fg-2 sm:text-base">{description}</p>}
      </div>
      {actions && <div className="flex shrink-0 gap-2">{actions}</div>}
    </motion.div>
  )
}

export function Alert({ tone = 'error', children }) {
  const styles = tone === 'error'
    ? 'border-ai/30 bg-ai/10 text-ai-text'
    : 'border-line-strong bg-raised text-fg-2'
  const Icon = tone === 'error' ? AlertTriangle : Info
  return (
    <div role={tone === 'error' ? 'alert' : 'note'} className={`flex items-start gap-2.5 rounded-lg border px-3.5 py-3 text-sm ${styles}`}>
      <Icon size={16} className="mt-0.5 shrink-0" />
      <div>{children}</div>
    </div>
  )
}

export function VerdictBadge({ isAI, children }) {
  const c = isAI ? VERDICT_COLORS.ai : VERDICT_COLORS.human
  const Icon = isAI ? AlertTriangle : CheckCircle2
  return (
    <span className="badge" style={{ color: c.text, background: `${c.mark}1f`, borderColor: `${c.mark}59` }}>
      <Icon size={13} /> {children}
    </span>
  )
}

/** One horizontal bar split into the AI share and the human/real share. */
export function SplitBar({ ai, aiLabel = 'AI-generated', otherLabel = 'Human-written' }) {
  const aiPct = ai * 100
  return (
    <div>
      <div className="mb-2 flex justify-between text-xs">
        <span className="flex items-center gap-1.5 text-fg-2">
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: VERDICT_COLORS.ai.mark }} />
          {aiLabel} <span className="tabular font-medium text-fg">{aiPct.toFixed(1)}%</span>
        </span>
        <span className="flex items-center gap-1.5 text-fg-2">
          <span className="tabular font-medium text-fg">{(100 - aiPct).toFixed(1)}%</span> {otherLabel}
          <span className="h-2.5 w-2.5 rounded-sm" style={{ background: VERDICT_COLORS.human.mark }} />
        </span>
      </div>
      <div className="flex h-2.5 w-full gap-0.5 overflow-hidden rounded-full">
        <motion.div initial={{ width: 0 }} animate={{ width: `${aiPct}%` }} transition={{ duration: 0.6, ease: 'easeOut' }}
          className="h-full" style={{ background: VERDICT_COLORS.ai.mark }} />
        <div className="h-full flex-1" style={{ background: VERDICT_COLORS.human.mark }} />
      </div>
    </div>
  )
}

/** Headline result: verdict badge, AI likelihood, split bar and run metadata. */
export function VerdictCard({ result, otherLabel }) {
  const isAI = result.ai_confidence > 0.5
  return (
    <div className="card p-5 sm:p-6">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-fg-3">Verdict</p>
          <VerdictBadge isAI={isAI}>{result.prediction}</VerdictBadge>
        </div>
        <div className="text-right">
          <p className="text-4xl font-semibold tracking-tight text-fg">{(result.ai_confidence * 100).toFixed(1)}%</p>
          <p className="text-xs text-fg-3">AI likelihood</p>
        </div>
      </div>
      <SplitBar ai={result.ai_confidence} otherLabel={otherLabel} />
      <dl className="mt-5 grid grid-cols-1 gap-3 border-t border-line pt-4 text-xs sm:grid-cols-[1fr_auto]">
        <div>
          <dt className="text-fg-3">Model</dt>
          <dd className="mt-0.5 text-fg-2">{result.model_used}</dd>
        </div>
        <div className="sm:text-right">
          <dt className="text-fg-3">Processing time</dt>
          <dd className="tabular mt-0.5 text-fg-2">{result.processing_time_ms} ms</dd>
        </div>
      </dl>
    </div>
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

export function LoadingState({ title, description }) {
  return (
    <div role="status" className="card flex flex-col items-center justify-center px-6 py-16 text-center">
      <Loader2 size={28} className="mb-4 animate-spin text-accent" />
      <p className="font-medium text-fg">{title}</p>
      {description && <p className="mt-1 text-sm text-fg-3">{description}</p>}
    </div>
  )
}

/** Two-model comparison used by both analysis pages. */
export function ModelComparison({ models }) {
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
            <div className="h-full rounded-full" style={{ width: `${ai * 100}%`, background: VERDICT_COLORS.ai.mark }} />
          </div>
          <p className="mt-2 text-xs text-fg-2">{prediction}</p>
        </div>
      ))}
    </div>
  )
}
