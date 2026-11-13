import { useEffect, useRef, useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, ScanSearch, Eye, BarChart3, Brain, Info, Loader2 } from 'lucide-react'
import { analyzeText, getErrorMessage, isCancel } from '../utils/api'
import { useTheme } from '../utils/theme'
import { usePersistentState } from '../utils/usePersistentState'
import {
  PageHeader, Alert, ApiOfflineBanner, VerdictCard, ReportActions, ReportHeader, CollapsibleCard,
  EmptyState, LoadingState, ModelComparison, fadeIn, scrollToOnMobile,
} from '../components/ui'

const MIN_CHARS = 20

const SAMPLES = [
  {
    label: 'AI-style sample',
    text: 'The implementation of blockchain technology in modern supply chain management systems facilitates unprecedented levels of traceability and transparency. Leveraging distributed ledger technology, organizations can seamlessly integrate real-time tracking mechanisms.',
  },
  {
    label: 'Human-style sample',
    text: 'I went to the market yesterday and honestly the tomatoes looked terrible. My neighbor said the same thing happened last week. Prices going up again is really frustrating for families like mine.',
  },
]

const alphaHex = (a) => Math.round(Math.min(Math.max(a, 0), 1) * 255).toString(16).padStart(2, '0')
// SHAP values are in probability units; ×100 gives percentage points of AI likelihood.
const pts = (v) => `${v >= 0 ? '+' : '−'}${Math.abs(v * 100).toFixed(1)} pts`

function TokenViz({ tokens }) {
  const { colors } = useTheme()
  const [active, setActive] = useState(null)
  const marks = { ai: colors.ai, human: colors.human }
  const labels = { ai: 'toward AI', human: 'toward human', neutral: 'little effect' }
  return (
    <div className="flex flex-wrap gap-1 text-sm leading-7">
      {tokens.map((t, i) => {
        const mark = marks[t.label]
        return (
          <span key={i} className="relative">
            <span tabIndex={0}
              onMouseEnter={() => setActive(i)} onMouseLeave={() => setActive(null)}
              onFocus={() => setActive(i)} onBlur={() => setActive(null)}
              className="inline-block cursor-default rounded px-1 outline-none transition-shadow focus-visible:ring-2 focus-visible:ring-accent"
              style={mark ? { background: `${mark}${alphaHex(0.15 + t.score * 0.5)}`, color: colors.fg } : { color: colors.fg2 }}>
              {t.token}
            </span>
            {active === i && (
              <span role="tooltip"
                className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded-md border border-line-strong bg-raised px-2.5 py-1.5 text-xs shadow-xl">
                <span className="tabular block font-semibold text-fg">{pts(t.shap)}</span>
                <span className="block text-fg-3">{labels[t.label]}</span>
              </span>
            )}
          </span>
        )
      })}
    </div>
  )
}

function TokenLegend() {
  const { colors } = useTheme()
  return (
    <div className="mb-4 flex flex-wrap gap-4 text-xs text-fg-2">
      {[['Pushes toward AI', colors.ai], ['Pushes toward human', colors.human], ['Little effect', null]].map(([label, c]) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-sm border border-line-strong" style={{ background: c ? `${c}aa` : 'transparent' }} />
          {label}
        </span>
      ))}
    </div>
  )
}

/** Diverging bars around a zero baseline: AI to the right, human to the left. */
function ShapChart({ values }) {
  const { colors } = useTheme()
  const max = Math.max(...values.map((v) => Math.abs(v.shap)), 1e-9)
  const cols = 'grid grid-cols-[minmax(0,6.5rem)_1fr_4.5rem] items-center gap-3'
  return (
    <div>
      <div className="space-y-1.5">
        {values.map((v, i) => {
          const towardAI = v.shap >= 0
          return (
            <div key={v.feature} className={`${cols} text-xs`}>
              <span className="truncate text-right text-fg-2" title={v.feature}>{v.feature}</span>
              <div className="relative h-5">
                <div className="absolute inset-y-0 left-1/2 w-px bg-line-strong" />
                <motion.div initial={{ width: 0 }} animate={{ width: `${(Math.abs(v.shap) / max) * 50}%` }}
                  transition={{ duration: 0.5, delay: i * 0.03 }}
                  className={`absolute inset-y-0.5 ${towardAI ? 'left-1/2 rounded-r' : 'right-1/2 rounded-l'}`}
                  style={{ background: towardAI ? colors.ai : colors.human }} />
              </div>
              <span className="tabular text-fg-2">{pts(v.shap)}</span>
            </div>
          )
        })}
      </div>
      <div className={`${cols} mt-3 text-[11px] text-fg-3`}>
        <span />
        <div className="flex justify-between"><span>← Human</span><span>AI →</span></div>
        <span />
      </div>
    </div>
  )
}

function buildSummary(text, r) {
  const confidence = (Math.max(r.ai_confidence, 1 - r.ai_confidence) * 100).toFixed(1)
  const lines = [
    `DeepFakeShield — text analysis (${new Date().toLocaleString()})`,
    `Verdict: ${r.prediction} (${confidence}% confidence)`,
    `AI likelihood: ${(r.ai_confidence * 100).toFixed(1)}%`,
    `Model: ${r.model_used}`,
  ]
  if (r.shap_values?.length) {
    lines.push('Top words (SHAP): ' + r.shap_values.slice(0, 6).map((f) => `${f.feature} ${pts(f.shap)}`).join(', '))
  }
  lines.push('', `Text: ${text.length > 300 ? text.slice(0, 300) + '…' : text}`)
  return lines.join('\n')
}

/** Shown only when printing (Print / save as PDF). */
function PrintReport({ text, result }) {
  return (
    <div className="hidden print:block">
      <ReportHeader title="Text analysis report" />
      <VerdictCard result={result} otherLabel="Human-written" />
      {result.shap_values?.length > 0 && (
        <div className="card mt-4 p-5">
          <p className="mb-3 text-sm font-semibold">Top SHAP features (percentage points of AI likelihood)</p>
          <ShapChart values={result.shap_values} />
        </div>
      )}
      <div className="card mt-4 p-5">
        <p className="mb-2 text-sm font-semibold">Analyzed text</p>
        <p className="whitespace-pre-wrap text-sm leading-relaxed text-fg-2">{text}</p>
      </div>
    </div>
  )
}

function Results({ text, result }) {
  const isEnsemble = result.roberta && result.linguaforge
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show" className="flex flex-col gap-4">
      <VerdictCard result={result} otherLabel="Human-written"
        actions={<ReportActions summary={buildSummary(text, result)} />} />

      {result.suspicious_phrases?.length > 0 && (
        <div className="card p-5 sm:p-6">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">Words pushing most toward AI</p>
          <div className="flex flex-wrap gap-2">
            {result.suspicious_phrases.map((p) => (
              <span key={p} className="badge border-line-strong bg-raised text-fg">
                <span className="h-1.5 w-1.5 rounded-full bg-ai" /> {p}
              </span>
            ))}
          </div>
        </div>
      )}

      {isEnsemble && (
        <CollapsibleCard title="Model comparison" icon={Brain}
          description={result.ensemble_weight
            ? `Final = RoBERTa × ${result.ensemble_weight.roberta} + LinguaForge-X × ${result.ensemble_weight.linguaforge}`
            : 'Final likelihood comes from a logistic-regression meta-learner over both models.'}>
          <ModelComparison models={[
            { name: 'RoBERTa', ai: result.roberta.ai_confidence, prediction: result.roberta.prediction, weight: result.ensemble_weight?.roberta },
            { name: 'LinguaForge-X', ai: result.linguaforge.ai_confidence, prediction: result.linguaforge.prediction, weight: result.ensemble_weight?.linguaforge },
          ]} />
        </CollapsibleCard>
      )}

      {result.explained === false ? (
        <Alert tone="info">
          The word-level explanation was skipped. Turn on <span className="text-fg">Explain the verdict</span> and analyze again
          to see which words drove it.
        </Alert>
      ) : (
        <>
          <CollapsibleCard title="Word-level explanation" icon={Eye}
            description="Each word is shaded by its SHAP value — how many percentage points it moved the AI likelihood. Hover or focus a word to see its value.">
            <TokenLegend />
            <TokenViz tokens={result.token_importance} />
          </CollapsibleCard>

          <CollapsibleCard title="Top SHAP features" icon={BarChart3}
            description="The words that moved the AI likelihood the most, in percentage points. Repeated words are summed; common words such as “the” or “of” are left out of this list.">
            <ShapChart values={result.shap_values} />
          </CollapsibleCard>
        </>
      )}
    </motion.div>
  )
}

export default function TextAnalysis() {
  const [text, setText] = usePersistentState('text.input', '')
  const [result, setResult] = usePersistentState('text.result', null)
  const [analyzedText, setAnalyzedText] = usePersistentState('text.analyzed', '')
  const [explain, setExplain] = usePersistentState('text.explain', true)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const abortRef = useRef(null)
  const resultsRef = useRef(null)

  const trimmed = text.trim()
  const words = trimmed ? trimmed.split(/\s+/).length : 0

  useEffect(() => () => abortRef.current?.abort(), [])

  const handleAnalyze = async () => {
    if (loading) return
    if (trimmed.length < MIN_CHARS) { setError(`Please enter at least ${MIN_CHARS} characters.`); return }
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true); setError(null); setResult(null)
    requestAnimationFrame(() => scrollToOnMobile(resultsRef))
    try {
      const data = await analyzeText(text, { explain, signal: controller.signal })
      setResult(data); setAnalyzedText(text)
      requestAnimationFrame(() => scrollToOnMobile(resultsRef))
    } catch (err) {
      if (!isCancel(err)) setError(getErrorMessage(err))
    } finally { setLoading(false) }
  }

  const onKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); handleAnalyze() }
  }

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="NLP analysis" title="Text analysis"
        description="Check whether a piece of text is AI-generated. Every verdict can include a SHAP explanation of the words that drove it." />
      <ApiOfflineBanner />

      {/* Phones: input → results → info.  Desktop: input and info on the left, results on the right. */}
      <div className="grid grid-cols-1 gap-6 print:hidden lg:grid-cols-12 lg:grid-rows-[auto_1fr] lg:items-start">
        <div className="card p-5 sm:p-6 lg:col-span-5">
          <div className="mb-3 flex items-center justify-between gap-3">
            <label htmlFor="text-input" className="text-sm font-semibold text-fg">Text to analyze</label>
            <span className="tabular text-xs text-fg-3">{words} words · {text.length} chars</span>
          </div>
          <textarea id="text-input" rows={12} value={text} onKeyDown={onKeyDown}
            onChange={(e) => { setText(e.target.value); setError(null) }}
            className="input min-h-[14rem] resize-y leading-relaxed"
            placeholder="Paste an article, social media post or paragraph…" />
          <div className="mt-3 flex flex-wrap items-center gap-2">
            <span className="text-xs text-fg-3">Try an example:</span>
            {SAMPLES.map((s) => (
              <button key={s.label} type="button" className="btn-secondary btn-sm"
                onClick={() => { setText(s.text); setError(null) }}>{s.label}</button>
            ))}
          </div>

          <label className="mt-4 flex cursor-pointer items-start gap-3 rounded-lg border border-line bg-raised px-3.5 py-3">
            <input type="checkbox" checked={explain} onChange={(e) => setExplain(e.target.checked)}
              className="mt-0.5 h-4 w-4 shrink-0 accent-[rgb(var(--accent))]" />
            <span className="text-sm">
              <span className="font-medium text-fg">Explain the verdict</span>
              <span className="block text-xs text-fg-3">Shows which words drove it (SHAP). Turn off for a faster result on large models.</span>
            </span>
          </label>

          {error && <div className="mt-4"><Alert>{error}</Alert></div>}
          <div className="mt-5 flex items-center gap-3 border-t border-line pt-4">
            <p className="hidden text-xs text-fg-3 sm:block">
              <kbd className="rounded border border-line-strong bg-raised px-1.5 py-0.5 font-sans text-[11px]">Ctrl</kbd> +{' '}
              <kbd className="rounded border border-line-strong bg-raised px-1.5 py-0.5 font-sans text-[11px]">Enter</kbd> to analyze
            </p>
            <div className="ml-auto flex gap-2">
              {text && !loading && (
                <button type="button" className="btn-ghost" onClick={() => { setText(''); setResult(null); setError(null) }}>Clear</button>
              )}
              <button id="analyze-text-btn" type="button" onClick={handleAnalyze} disabled={loading || !trimmed}
                className="btn-primary">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><ScanSearch size={16} /> Analyze text</>}
              </button>
            </div>
          </div>
        </div>

        <div ref={resultsRef} className="scroll-mt-20 lg:col-span-7 lg:col-start-6 lg:row-span-2 lg:row-start-1" aria-live="polite">
          {loading ? (
            <LoadingState title="Analyzing text"
              description={explain ? 'Running the model and computing SHAP values…' : 'Running the model…'}
              onCancel={() => abortRef.current?.abort()} />
          ) : result ? (
            <Results text={analyzedText} result={result} />
          ) : (
            <EmptyState icon={FileText} title="No analysis yet"
              description="Paste some text or pick an example, then select Analyze text. The verdict and its explanation will appear here." />
          )}
        </div>

        <div className="card p-5 sm:p-6 lg:col-span-5">
          <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-fg">
            <Info size={15} className="text-fg-3" /> How detection works
          </p>
          <ul className="space-y-2 text-[13px] leading-relaxed text-fg-2">
            <li><span className="text-fg">Pre-trained detector (recommended):</span> a RoBERTa model trained on the RAID benchmark, once downloaded with <code className="font-mono text-xs text-fg">download_models.py</code>.</li>
            <li><span className="text-fg">Trained models:</span> the project's own RoBERTa (55%) + LinguaForge-X (45%) ensemble.</li>
            <li><span className="text-fg">No model:</span> text analysis is unavailable until one of the above is installed.</li>
            <li><span className="text-fg">Explanation:</span> SHAP values computed on whichever model produced the result.</li>
          </ul>
        </div>
      </div>

      {result && <PrintReport text={analyzedText} result={result} />}
    </main>
  )
}
