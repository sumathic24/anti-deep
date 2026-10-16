import { useState } from 'react'
import { motion } from 'framer-motion'
import { FileText, ScanSearch, Eye, BarChart3, Brain, Info, Loader2 } from 'lucide-react'
import { analyzeText, getErrorMessage } from '../utils/api'
import {
  PageHeader, Alert, VerdictCard, CollapsibleCard, EmptyState, LoadingState, ModelComparison,
  VERDICT_COLORS, fadeIn,
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

function TokenViz({ tokens }) {
  const [active, setActive] = useState(null)
  const marks = { ai: VERDICT_COLORS.ai.mark, human: VERDICT_COLORS.human.mark }
  const labels = { ai: 'Pushes toward AI', human: 'Pushes toward human', neutral: 'Little effect' }
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
              style={mark ? { background: `${mark}${alphaHex(0.15 + t.score * 0.5)}`, color: '#e6eaf2' } : { color: '#a3acbf' }}>
              {t.token}
            </span>
            {active === i && (
              <span role="tooltip"
                className="pointer-events-none absolute bottom-full left-1/2 z-20 mb-1.5 -translate-x-1/2 whitespace-nowrap rounded-md border border-line-strong bg-raised px-2.5 py-1.5 text-xs shadow-xl">
                <span className="tabular block font-semibold text-fg">{t.shap >= 0 ? '+' : ''}{t.shap.toFixed(4)}</span>
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
  return (
    <div className="mb-4 flex flex-wrap gap-4 text-xs text-fg-2">
      {[['Pushes toward AI', VERDICT_COLORS.ai.mark], ['Pushes toward human', VERDICT_COLORS.human.mark], ['Little effect', null]].map(([label, c]) => (
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
  const max = Math.max(...values.map((v) => Math.abs(v.shap)), 1e-9)
  const cols = 'grid grid-cols-[minmax(0,6.5rem)_1fr_4rem] items-center gap-3'
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
                  style={{ background: towardAI ? VERDICT_COLORS.ai.mark : VERDICT_COLORS.human.mark }} />
              </div>
              <span className="tabular text-fg-2">{towardAI ? '+' : ''}{v.shap.toFixed(3)}</span>
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

function Results({ result }) {
  const isEnsemble = result.roberta && result.linguaforge
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show" className="flex flex-col gap-4">
      <VerdictCard result={result} otherLabel="Human-written" />

      {result.suspicious_phrases?.length > 0 && (
        <div className="card p-5 sm:p-6">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">Words pushing most toward AI</p>
          <div className="flex flex-wrap gap-2">
            {result.suspicious_phrases.map((p) => (
              <span key={p} className="badge border-line-strong bg-raised text-fg">
                <span className="h-1.5 w-1.5 rounded-full" style={{ background: VERDICT_COLORS.ai.mark }} /> {p}
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

      <CollapsibleCard title="Word-level explanation" icon={Eye}
        description="Each word is shaded by its SHAP value — how much it pushed this prediction toward AI or toward human. Hover or focus a word to see its value.">
        <TokenLegend />
        <TokenViz tokens={result.token_importance} />
      </CollapsibleCard>

      <CollapsibleCard title="Top SHAP features" icon={BarChart3}
        description="The words with the largest effect on this prediction. Repeated words are summed.">
        <ShapChart values={result.shap_values} />
      </CollapsibleCard>
    </motion.div>
  )
}

export default function TextAnalysis() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  const trimmed = text.trim()
  const words = trimmed ? trimmed.split(/\s+/).length : 0

  const handleAnalyze = async () => {
    if (loading) return
    if (trimmed.length < MIN_CHARS) { setError(`Please enter at least ${MIN_CHARS} characters.`); return }
    setLoading(true); setError(null); setResult(null)
    try {
      setResult(await analyzeText(text))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally { setLoading(false) }
  }

  const onKeyDown = (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); handleAnalyze() }
  }

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="NLP analysis" title="Text analysis"
        description="Check whether a piece of text is AI-generated. Every verdict includes a SHAP explanation of the words that drove it." />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Input */}
        <div className="flex flex-col gap-4 lg:col-span-5">
          <div className="card p-5 sm:p-6">
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

          <div className="card p-5 sm:p-6">
            <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-fg">
              <Info size={15} className="text-fg-3" /> How detection works
            </p>
            <ul className="space-y-2 text-[13px] leading-relaxed text-fg-2">
              <li><span className="text-fg">Pre-trained detector (recommended):</span> a RoBERTa model trained on the RAID benchmark, once downloaded with <code className="font-mono text-xs text-fg">download_models.py</code>.</li>
              <li><span className="text-fg">Trained models:</span> the project's own RoBERTa (55%) + LinguaForge-X (45%) ensemble.</li>
              <li><span className="text-fg">Fallback:</span> a small bag-of-words model when neither is available.</li>
              <li><span className="text-fg">Explanation:</span> SHAP values computed on whichever model produced the result — shown with every verdict.</li>
            </ul>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-7" aria-live="polite">
          {loading ? (
            <LoadingState title="Analyzing text" description="Running the model and computing SHAP values…" />
          ) : result ? (
            <Results result={result} />
          ) : (
            <EmptyState icon={FileText} title="No analysis yet"
              description="Paste some text or pick an example, then select Analyze text. The verdict and its explanation will appear here." />
          )}
        </div>
      </div>
    </main>
  )
}
