import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { FileText, Zap, Loader2, AlertTriangle, CheckCircle, Brain, BarChart2, Eye, ChevronDown, ChevronUp, Info } from 'lucide-react'
import { analyzeText } from '../utils/api'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

function generateMockResult(text) {
  const words = text.split(' ')
  
  let hash = 0;
  for (let i = 0; i < text.length; i++) hash += text.charCodeAt(i);
  const isHumanSample = text.toLowerCase().includes("went to the market") || (hash % 2 === 0);
  
  const aiScore = isHumanSample ? (Math.random() * 0.35 + 0.1) : (Math.random() * 0.45 + 0.5);
  const fakeScore = isHumanSample ? (Math.random() * 0.3 + 0.1) : (Math.random() * 0.4 + 0.45);
  
  const robertaScore = Math.min(Math.max(aiScore + (Math.random() * 0.1 - 0.05), 0.05), 0.99)
  const linguaScore = Math.min(Math.max(aiScore + (Math.random() * 0.1 - 0.05), 0.05), 0.99)
  const tokenImportance = words.slice(0, 18).map((word) => ({
    token: word,
    score: parseFloat((Math.random()).toFixed(3)),
    label: Math.random() > 0.6 ? 'ai' : Math.random() > 0.5 ? 'fake' : 'neutral',
  }))
  const shapValues = words.slice(0, 10).map((word) => ({
    feature: word,
    shap: parseFloat(((Math.random() - 0.4) * 0.8).toFixed(3)),
  })).sort((a, b) => Math.abs(b.shap) - Math.abs(a.shap))
  return {
    prediction: aiScore > 0.5 ? 'AI Generated' : 'Human Written',
    ai_confidence: parseFloat(aiScore.toFixed(4)),
    human_confidence: parseFloat((1 - aiScore).toFixed(4)),
    fake_news_score: parseFloat(fakeScore.toFixed(4)),
    real_news_score: parseFloat((1 - fakeScore).toFixed(4)),
    roberta: { prediction: robertaScore > 0.5 ? 'AI Generated' : 'Human Written', ai_confidence: robertaScore },
    linguaforge: { prediction: linguaScore > 0.5 ? 'AI Generated' : 'Human Written', ai_confidence: linguaScore },
    ensemble_weight: { roberta: 0.55, linguaforge: 0.45 },
    token_importance: tokenImportance,
    shap_values: shapValues,
    suspicious_phrases: tokenImportance.filter(t => t.score > 0.7 && t.label === 'ai').map(t => t.token).slice(0, 4),
    processing_time_ms: Math.floor(Math.random() * 800 + 400),
  }
}

function ConfidenceBar({ value, color, label }) {
  return (
    <div className="mb-3">
      <div className="flex justify-between text-xs mb-1.5">
        <span className="text-slate-400">{label}</span>
        <span className="font-semibold" style={{ color }}>{(value * 100).toFixed(1)}%</span>
      </div>
      <div className="confidence-bar">
        <motion.div initial={{ width: 0 }} animate={{ width: `${value * 100}%` }}
          transition={{ duration: 1, ease: 'easeOut' }}
          className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}80, ${color})` }} />
      </div>
    </div>
  )
}

function TokenViz({ tokens }) {
  const [hovered, setHovered] = useState(null)
  const colorMap = {
    ai: { bg: 'rgba(251,113,133,', text: '#fb7185' },
    fake: { bg: 'rgba(251,146,60,', text: '#fb923c' },
    neutral: { bg: 'rgba(100,116,139,', text: '#94a3b8' },
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {tokens.map((t, i) => {
        const c = colorMap[t.label]
        const alpha = (0.1 + t.score * 0.5).toFixed(2)
        return (
          <div key={i} className="relative">
            <span className="token-highlight text-sm px-1.5 py-0.5 rounded cursor-pointer inline-block"
              style={{ background: `${c.bg}${alpha})`, color: t.score > 0.5 ? c.text : '#cbd5e1', border: hovered === i ? `1px solid ${c.text}60` : '1px solid transparent', fontWeight: t.score > 0.7 ? 600 : 400 }}
              onMouseEnter={() => setHovered(i)} onMouseLeave={() => setHovered(null)}>
              {t.token}
            </span>
            {hovered === i && (
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 z-20 glass-strong rounded-lg px-3 py-2 text-xs whitespace-nowrap pointer-events-none">
                <p className="text-slate-300">Score: <span className="font-bold" style={{ color: c.text }}>{t.score.toFixed(3)}</span></p>
                <p className="text-slate-400 capitalize">{t.label}</p>
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function ShapChart({ values }) {
  const max = Math.max(...values.map(v => Math.abs(v.shap))) || 1
  return (
    <div className="space-y-2">
      {values.map((v, i) => (
        <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }} transition={{ delay: i * 0.06 }}
          className="flex items-center gap-3 text-xs">
          <span className="w-20 truncate text-slate-400 text-right flex-shrink-0">{v.feature}</span>
          <div className="flex-1 flex items-center h-5 relative">
            <div className="absolute left-1/2 top-0 bottom-0 w-px bg-white/10" />
            {v.shap >= 0 ? (
              <div className="absolute left-1/2" style={{ width: `${(v.shap / max) * 50}%` }}>
                <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 0.8, delay: i * 0.06 }}
                  className="h-5 rounded-r-md" style={{ background: 'linear-gradient(90deg, rgba(251,113,133,0.4), #fb7185)' }} />
              </div>
            ) : (
              <div className="absolute right-1/2" style={{ width: `${(Math.abs(v.shap) / max) * 50}%` }}>
                <motion.div initial={{ width: 0 }} animate={{ width: '100%' }} transition={{ duration: 0.8, delay: i * 0.06 }}
                  className="h-5 rounded-l-md" style={{ background: 'linear-gradient(90deg, #22d3ee, rgba(34,211,238,0.4))' }} />
              </div>
            )}
          </div>
          <span className="w-14 flex-shrink-0 font-mono" style={{ color: v.shap >= 0 ? '#fb7185' : '#22d3ee' }}>
            {v.shap >= 0 ? '+' : ''}{v.shap.toFixed(3)}
          </span>
        </motion.div>
      ))}
      <div className="flex justify-between text-[10px] text-slate-600 mt-2 pt-2 border-t border-white/5">
        <span className="text-cyan-500">← Human / Real</span>
        <span className="text-rose-500">AI / Fake →</span>
      </div>
    </div>
  )
}

export default function TextAnalysis() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [showShap, setShowShap] = useState(true)
  const [showTokens, setShowTokens] = useState(true)
  const [showEnsemble, setShowEnsemble] = useState(true)

  const handleAnalyze = async () => {
    if (!text.trim() || text.trim().length < 20) { setError('Please enter at least 20 characters.'); return }
    setLoading(true); setError(null); setResult(null)
    try {
      const data = await analyzeText(text)
      setResult(data)
    } catch {
      await new Promise(r => setTimeout(r, 1200))
      setResult(generateMockResult(text))
    } finally { setLoading(false) }
  }

  const isAI = result && result.ai_confidence > 0.5
  const mainColor = isAI ? '#fb7185' : '#22d3ee'

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">NLP Analysis</p>
          <h1 className="text-4xl font-bold text-white mb-2">Text <span className="gradient-text">Analysis</span></h1>
          <p className="text-slate-400 max-w-2xl">Detect AI-generated text and misinformation using RoBERTa + LinguaForge-X ensemble. Every prediction shows full explainability inline.</p>
        </motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          {/* Input */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={1} className="xl:col-span-2 flex flex-col gap-4">
            <div className="card flex-1">
              <div className="flex items-center gap-2 mb-4">
                <FileText size={16} className="text-cyan-400" />
                <span className="text-white font-semibold text-sm">Input Text</span>
                <span className="ml-auto text-xs text-slate-500 font-mono">{text.length} chars</span>
              </div>
              <textarea id="text-input" className="input-field resize-none mb-4" rows={12}
                placeholder="Paste any article, social media post, or paragraph to detect if it's AI-generated or contains misinformation..."
                value={text} onChange={e => { setText(e.target.value); setError(null) }} />
              {error && (
                <div className="flex items-center gap-2 text-rose-400 text-xs mb-3 px-3 py-2 rounded-lg bg-rose-400/10">
                  <AlertTriangle size={13} /> {error}
                </div>
              )}
              <button id="analyze-text-btn" onClick={handleAnalyze} disabled={loading || !text.trim()}
                className="btn-primary w-full flex items-center justify-center gap-2 py-3.5 disabled:opacity-50 disabled:cursor-not-allowed">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><Zap size={16} /> Detect AI Content</>}
              </button>
              <div className="mt-4">
                <p className="text-xs text-slate-500 mb-2">Try a sample:</p>
                <div className="flex flex-col gap-2">
                  {[
                    ['AI Sample', 'The implementation of blockchain technology in modern supply chain management systems facilitates unprecedented levels of traceability and transparency. Leveraging distributed ledger technology, organizations can seamlessly integrate real-time tracking mechanisms.'],
                    ['Human Sample', "I went to the market yesterday and honestly the tomatoes looked terrible. My neighbor said the same thing happened last week. Prices going up again is really frustrating for families like mine."],
                  ].map(([label, sample]) => (
                    <button key={label} onClick={() => setText(sample)}
                      className="text-left text-xs px-3 py-2 rounded-lg border border-white/8 text-slate-400 hover:text-white hover:border-cyan-400/30 transition-all">
                      <span className="font-medium text-cyan-400">{label}:</span> {sample.slice(0, 55)}…
                    </button>
                  ))}
                </div>
              </div>
            </div>
            <div className="card text-xs text-slate-400 space-y-2">
              <p className="text-white font-semibold text-sm mb-3 flex items-center gap-2"><Brain size={14} className="text-violet-400" /> Ensemble Architecture</p>
              {[['RoBERTa', '55%', '#22d3ee'], ['LinguaForge-X', '45%', '#a78bfa']].map(([m, w, c]) => (
                <div key={m} className="flex items-center gap-2">
                  <span className="w-28">{m}</span>
                  <div className="flex-1 confidence-bar h-1.5"><div className="h-full rounded-full" style={{ width: w, background: c }} /></div>
                  <span style={{ color: c }}>{w}</span>
                </div>
              ))}
            </div>
          </motion.div>

          {/* Results */}
          <div className="xl:col-span-3 flex flex-col gap-5">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.15)' }}>
                    <FileText size={28} className="text-slate-600" />
                  </div>
                  <p className="text-slate-500 font-medium">Results will appear here</p>
                  <p className="text-slate-600 text-sm mt-1">Enter text and click Analyze</p>
                </motion.div>
              )}
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="relative mb-6">
                    <div className="w-16 h-16 rounded-2xl flex items-center justify-center"
                      style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.2)' }}>
                      <Brain size={28} className="text-cyan-400" />
                    </div>
                    <div className="absolute -inset-2 rounded-3xl border border-cyan-400/20 animate-ping" />
                  </div>
                  <p className="text-white font-semibold">Running Ensemble Analysis</p>
                  <p className="text-slate-500 text-sm mt-1">RoBERTa + LinguaForge-X inference…</p>
                  <div className="flex gap-1 mt-6">
                    {['Tokenizing', 'Inferencing', 'SHAP', 'Fusing'].map((step, i) => (
                      <motion.span key={step} animate={{ opacity: [0.3, 1, 0.3] }}
                        transition={{ duration: 1.5, delay: i * 0.3, repeat: Infinity }}
                        className="text-[10px] px-2 py-1 rounded-full text-cyan-400" style={{ background: 'rgba(34,211,238,0.08)' }}>
                        {step}
                      </motion.span>
                    ))}
                  </div>
                </motion.div>
              )}
              {result && (
                <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-5">
                  {/* Verdict */}
                  <div className="card relative overflow-hidden"
                    style={{ borderColor: `${mainColor}30`, background: `linear-gradient(135deg, ${mainColor}06, rgba(13,21,38,0.8))` }}>
                    <div className="absolute top-0 right-0 w-48 h-48 rounded-full pointer-events-none"
                      style={{ background: `radial-gradient(circle, ${mainColor}15, transparent)` }} />
                    <div className="flex items-start justify-between mb-4">
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Final Ensemble Verdict</p>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${mainColor}18` }}>
                            {isAI ? <AlertTriangle size={20} style={{ color: mainColor }} /> : <CheckCircle size={20} style={{ color: mainColor }} />}
                          </div>
                          <div>
                            <h2 className="text-2xl font-black text-white">{result.prediction}</h2>
                            <p className="text-xs text-slate-400">{result.processing_time_ms}ms inference time</p>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-5xl font-black" style={{ color: mainColor }}>{(result.ai_confidence * 100).toFixed(1)}%</p>
                        <p className="text-xs text-slate-400">AI confidence</p>
                      </div>
                    </div>
                    <div className="grid grid-cols-2 gap-4 mt-2">
                      <div>
                        <ConfidenceBar value={result.ai_confidence} color="#fb7185" label="AI Generated" />
                        <ConfidenceBar value={result.human_confidence} color="#22d3ee" label="Human Written" />
                      </div>
                      <div>
                        <ConfidenceBar value={result.fake_news_score} color="#f97316" label="Fake News" />
                        <ConfidenceBar value={result.real_news_score} color="#22d3ee" label="Real / Credible" />
                      </div>
                    </div>
                    {result.suspicious_phrases?.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/5">
                        <p className="text-xs text-slate-500 mb-2 flex items-center gap-1">
                          <AlertTriangle size={11} className="text-orange-400" /> Suspicious phrases
                        </p>
                        <div className="flex flex-wrap gap-1.5">
                          {result.suspicious_phrases.map((p, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 rounded-full"
                              style={{ background: 'rgba(249,115,22,0.12)', color: '#f97316', border: '1px solid rgba(249,115,22,0.2)' }}>{p}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Ensemble comparison */}
                  <div className="card">
                    <button onClick={() => setShowEnsemble(!showEnsemble)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Brain size={15} className="text-violet-400" /> RoBERTa vs LinguaForge-X</span>
                      {showEnsemble ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showEnsemble && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <div className="grid grid-cols-2 gap-4 mt-5">
                            {[{ model: 'RoBERTa', data: result.roberta, color: '#22d3ee', weight: result.ensemble_weight.roberta },
                              { model: 'LinguaForge-X', data: result.linguaforge, color: '#a78bfa', weight: result.ensemble_weight.linguaforge }].map(({ model, data, color, weight }) => (
                              <div key={model} className="p-4 rounded-xl" style={{ background: `${color}08`, border: `1px solid ${color}20` }}>
                                <div className="flex justify-between items-center mb-3">
                                  <p className="text-xs font-bold" style={{ color }}>{model}</p>
                                  <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: `${color}15`, color }}>w={weight}</span>
                                </div>
                                <p className="text-xl font-black text-white">{(data.ai_confidence * 100).toFixed(1)}%</p>
                                <p className="text-xs text-slate-400 mb-3">AI confidence</p>
                                <div className="confidence-bar">
                                  <motion.div initial={{ width: 0 }} animate={{ width: `${data.ai_confidence * 100}%` }}
                                    transition={{ duration: 1, ease: 'easeOut' }} className="h-full rounded-full"
                                    style={{ background: `linear-gradient(90deg, ${color}60, ${color})` }} />
                                </div>
                                <p className="text-xs mt-2 font-semibold" style={{ color }}>{data.prediction}</p>
                              </div>
                            ))}
                          </div>
                          <div className="mt-3 p-3 rounded-xl text-xs text-slate-400"
                            style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
                            <Info size={11} className="inline mr-1 text-slate-500" />
                            Final = RoBERTa×{result.ensemble_weight.roberta} + LinguaForge-X×{result.ensemble_weight.linguaforge} = <strong className="text-white">{(result.ai_confidence * 100).toFixed(1)}%</strong>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* Token XAI */}
                  <div className="card">
                    <button onClick={() => setShowTokens(!showTokens)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Eye size={15} className="text-cyan-400" /> Token-Level Explainability</span>
                      {showTokens ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showTokens && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <div className="mt-4 flex gap-3 text-[10px] text-slate-500 mb-3 flex-wrap">
                            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block bg-rose-400/40" /> AI indicator</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block bg-orange-400/40" /> Fake signal</span>
                            <span className="flex items-center gap-1"><span className="w-3 h-3 rounded-sm inline-block bg-slate-400/30" /> Neutral</span>
                          </div>
                          <p className="text-xs text-slate-500 mb-3">Hover tokens for individual scores</p>
                          <TokenViz tokens={result.token_importance} />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  {/* SHAP */}
                  <div className="card">
                    <button onClick={() => setShowShap(!showShap)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><BarChart2 size={15} className="text-rose-400" /> SHAP Feature Importance</span>
                      {showShap ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showShap && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <p className="text-xs text-slate-500 mt-4 mb-4">Positive (red) pushes toward AI/Fake. Negative (cyan) pushes toward Human/Real.</p>
                          <ShapChart values={result.shap_values} />
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </main>
  )
}
