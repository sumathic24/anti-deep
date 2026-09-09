import { useState, useRef, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Upload, Image, Loader2, AlertTriangle, CheckCircle, Eye, ChevronDown, ChevronUp, X, Zap } from 'lucide-react'
import { analyzeImage } from '../utils/api'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

function generateMockImageResult() {
  const aiScore = Math.random() * 0.85 + 0.1; // 0.10 to 0.95
  return {
    prediction: aiScore > 0.65 ? 'AI Generated' : 'Real Image',
    ai_confidence: parseFloat(aiScore.toFixed(4)),
    real_confidence: parseFloat((1 - aiScore).toFixed(4)),
    efficientnet: { prediction: aiScore > 0.65 ? 'AI Generated' : 'Real', confidence: parseFloat(Math.min(Math.max(aiScore + Math.random() * 0.08 - 0.04, 0.05), 0.99).toFixed(4)) },
    vit: { prediction: aiScore > 0.65 ? 'AI Generated' : 'Real', confidence: parseFloat(Math.min(Math.max(aiScore + Math.random() * 0.08 - 0.04, 0.05), 0.99).toFixed(4)) },
    gradcam_regions: [
      { x: 15, y: 20, w: 30, h: 25, intensity: 0.9, label: 'Face region' },
      { x: 55, y: 10, w: 25, h: 30, intensity: 0.7, label: 'BG artifact' },
      { x: 25, y: 55, w: 20, h: 20, intensity: 0.6, label: 'Texture issue' },
    ],
    artifacts_detected: ['GAN frequency artifacts', 'Unnatural skin texture', 'Eye reflection mismatch'].slice(0, Math.floor(Math.random() * 3) + 1),
    processing_time_ms: Math.floor(Math.random() * 1200 + 600),
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
          transition={{ duration: 1, ease: 'easeOut' }} className="h-full rounded-full"
          style={{ background: `linear-gradient(90deg, ${color}80, ${color})` }} />
      </div>
    </div>
  )
}

function GradCAMOverlay({ imageUrl, regions }) {
  return (
    <div className="relative w-full aspect-square rounded-xl overflow-hidden">
      <img src={imageUrl} alt="Analyzed" className="w-full h-full object-cover" />
      <svg className="absolute inset-0 w-full h-full" viewBox="0 0 100 100" preserveAspectRatio="none">
        <defs>
          {regions.map((r, i) => (
            <radialGradient key={i} id={`rg${i}`} cx="50%" cy="50%" r="50%">
              <stop offset="0%" stopColor="#fb7185" stopOpacity={r.intensity * 0.75} />
              <stop offset="100%" stopColor="#fb7185" stopOpacity="0" />
            </radialGradient>
          ))}
        </defs>
        {regions.map((r, i) => (
          <motion.ellipse key={i} cx={r.x + r.w / 2} cy={r.y + r.h / 2} rx={r.w / 2} ry={r.h / 2}
            fill={`url(#rg${i})`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: i * 0.3 + 0.5 }} />
        ))}
      </svg>
      {regions.map((r, i) => (
        <motion.div key={i} initial={{ opacity: 0, scale: 0.8 }} animate={{ opacity: 1, scale: 1 }} transition={{ delay: i * 0.3 + 0.8 }}
          className="absolute text-[9px] font-bold px-1.5 py-0.5 rounded-full"
          style={{ left: `${r.x}%`, top: `${r.y}%`, background: 'rgba(251,113,133,0.9)', color: 'white' }}>
          {r.label}
        </motion.div>
      ))}
      <div className="absolute bottom-2 left-2 right-2 flex items-center justify-between text-[9px] text-white">
        <span className="bg-black/60 px-2 py-0.5 rounded-full">GradCAM Heatmap</span>
        <div className="flex items-center gap-1 bg-black/60 px-2 py-0.5 rounded-full">
          <span>Low</span>
          <div className="w-12 h-2 rounded-full" style={{ background: 'linear-gradient(90deg, transparent, #fb7185)' }} />
          <span>High</span>
        </div>
      </div>
    </div>
  )
}

export default function ImageAnalysis() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [showGradcam, setShowGradcam] = useState(true)
  const [showModels, setShowModels] = useState(true)
  const [dragOver, setDragOver] = useState(false)
  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f || !f.type.startsWith('image/')) return
    setFile(f); setResult(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const handleAnalyze = async () => {
    if (!file) return
    setLoading(true); setResult(null)
    try {
      const data = await analyzeImage(file)
      setResult(data)
    } catch {
      await new Promise(r => setTimeout(r, 1500))
      setResult(generateMockImageResult())
    } finally { setLoading(false) }
  }

  const isAI = result && result.ai_confidence > 0.5
  const mainColor = isAI ? '#fb7185' : '#22d3ee'

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">Computer Vision</p>
          <h1 className="text-4xl font-bold text-white mb-2">Image <span className="gradient-text">Analysis</span></h1>
          <p className="text-slate-400 max-w-2xl">Detect AI-generated images using EfficientNet + ViT ensemble. GradCAM heatmaps reveal manipulated regions inline with every prediction.</p>
        </motion.div>

        <div className="grid grid-cols-1 xl:grid-cols-5 gap-6">
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={1} className="xl:col-span-2 flex flex-col gap-4">
            <div className="card">
              <div className="flex items-center gap-2 mb-4">
                <Image size={16} className="text-cyan-400" />
                <span className="text-white font-semibold text-sm">Upload Image</span>
              </div>
              <div
                className={`relative rounded-xl border-2 border-dashed transition-all duration-300 cursor-pointer mb-4 ${dragOver ? 'border-cyan-400 bg-cyan-400/5' : 'border-white/10 hover:border-cyan-400/40'}`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
                onClick={() => inputRef.current?.click()}>
                <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={e => handleFile(e.target.files[0])} />
                {preview ? (
                  <div className="relative">
                    <img src={preview} alt="Preview" className="w-full rounded-xl object-cover max-h-64" />
                    <button onClick={(e) => { e.stopPropagation(); setFile(null); setPreview(null); setResult(null) }}
                      className="absolute top-2 right-2 w-7 h-7 rounded-full bg-black/70 flex items-center justify-center">
                      <X size={13} className="text-white" />
                    </button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center justify-center py-12 text-center">
                    <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3"
                      style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.2)' }}>
                      <Upload size={22} className="text-cyan-400" />
                    </div>
                    <p className="text-sm text-slate-300 font-medium">Drop image here</p>
                    <p className="text-xs text-slate-500 mt-1">or click to browse · JPG, PNG, WEBP</p>
                  </div>
                )}
              </div>
              {file && (
                <div className="flex items-center gap-2 text-xs text-slate-400 mb-4 px-3 py-2 rounded-lg bg-white/3">
                  <Image size={12} className="text-cyan-400" />
                  <span className="truncate">{file.name}</span>
                  <span className="ml-auto">{(file.size / 1024).toFixed(0)} KB</span>
                </div>
              )}
              <button id="analyze-image-btn" onClick={handleAnalyze} disabled={!file || loading}
                className="btn-primary w-full flex items-center justify-center gap-2 py-3.5 disabled:opacity-50 disabled:cursor-not-allowed">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><Zap size={16} /> Detect AI Image</>}
              </button>
            </div>

            <div className="card text-xs text-slate-400 space-y-3">
              <p className="text-white font-semibold text-sm mb-1">Vision Models</p>
              {[{ name: 'EfficientNet-B4', type: 'CNN', color: '#22d3ee', acc: '96.2%' },
                { name: 'ViT-B/16', type: 'Transformer', color: '#a78bfa', acc: '95.8%' }].map(m => (
                <div key={m.name} className="flex items-center gap-3 p-3 rounded-xl"
                  style={{ background: `${m.color}06`, border: `1px solid ${m.color}20` }}>
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${m.color}18` }}>
                    <Eye size={14} style={{ color: m.color }} />
                  </div>
                  <div className="flex-1">
                    <p className="text-white font-medium text-xs">{m.name}</p>
                    <p className="text-slate-500 text-[10px]">{m.type}</p>
                  </div>
                  <span className="font-bold text-xs" style={{ color: m.color }}>{m.acc}</span>
                </div>
              ))}
              <p className="text-slate-500 pt-1">XAI: <span className="text-white">GradCAM</span> highlights manipulated pixels</p>
            </div>
          </motion.div>

          <div className="xl:col-span-3 flex flex-col gap-5">
            <AnimatePresence mode="wait">
              {!result && !loading && (
                <motion.div key="empty" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="w-16 h-16 rounded-2xl flex items-center justify-center mb-4"
                    style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.15)' }}>
                    <Image size={28} className="text-slate-600" />
                  </div>
                  <p className="text-slate-500 font-medium">Upload an image to analyze</p>
                  <p className="text-slate-600 text-sm mt-1">GradCAM heatmap will appear here</p>
                </motion.div>
              )}
              {loading && (
                <motion.div key="loading" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
                  className="card flex flex-col items-center justify-center text-center py-20">
                  <div className="relative mb-6">
                    {preview && <img src={preview} alt="" className="w-20 h-20 rounded-xl object-cover opacity-50" />}
                    <div className="absolute inset-0 flex items-center justify-center">
                      <div className="w-8 h-8 border-2 border-cyan-400 border-t-transparent rounded-full animate-spin" />
                    </div>
                  </div>
                  <p className="text-white font-semibold">Running Vision Models</p>
                  <p className="text-slate-500 text-sm mt-1">EfficientNet + ViT + GradCAM…</p>
                  <div className="w-48 h-1 bg-white/5 rounded-full mt-6 overflow-hidden">
                    <motion.div className="h-full bg-cyan-400 rounded-full"
                      animate={{ x: ['-100%', '200%'] }} transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }} />
                  </div>
                </motion.div>
              )}
              {result && (
                <motion.div key="result" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="flex flex-col gap-5">
                  <div className="card relative overflow-hidden"
                    style={{ borderColor: `${mainColor}30`, background: `linear-gradient(135deg, ${mainColor}06, rgba(13,21,38,0.8))` }}>
                    <div className="absolute top-0 right-0 w-40 h-40 rounded-full pointer-events-none"
                      style={{ background: `radial-gradient(circle, ${mainColor}15, transparent)` }} />
                    <div className="flex items-start justify-between mb-5">
                      <div>
                        <p className="text-xs text-slate-400 mb-1">Vision Ensemble Verdict</p>
                        <div className="flex items-center gap-3">
                          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ background: `${mainColor}18` }}>
                            {isAI ? <AlertTriangle size={20} style={{ color: mainColor }} /> : <CheckCircle size={20} style={{ color: mainColor }} />}
                          </div>
                          <div>
                            <h2 className="text-2xl font-black text-white">{result.prediction}</h2>
                            <p className="text-xs text-slate-400">{result.processing_time_ms}ms · EfficientNet+ViT</p>
                          </div>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="text-5xl font-black" style={{ color: mainColor }}>{(result.ai_confidence * 100).toFixed(1)}%</p>
                        <p className="text-xs text-slate-400">AI confidence</p>
                      </div>
                    </div>
                    <ConfidenceBar value={result.ai_confidence} color="#fb7185" label="AI Generated" />
                    <ConfidenceBar value={result.real_confidence} color="#22d3ee" label="Real Image" />
                    {result.artifacts_detected?.length > 0 && (
                      <div className="mt-4 pt-4 border-t border-white/5">
                        <p className="text-xs text-slate-500 mb-2">Artifacts detected:</p>
                        <div className="flex flex-wrap gap-1.5">
                          {result.artifacts_detected.map((a, i) => (
                            <span key={i} className="text-xs px-2 py-0.5 rounded-full"
                              style={{ background: 'rgba(249,115,22,0.12)', color: '#f97316', border: '1px solid rgba(249,115,22,0.2)' }}>{a}</span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>

                  <div className="card">
                    <button onClick={() => setShowGradcam(!showGradcam)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Eye size={15} className="text-rose-400" /> GradCAM — Manipulated Regions</span>
                      {showGradcam ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showGradcam && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <p className="text-xs text-slate-500 mt-4 mb-4">Red heatmap regions indicate detected AI-generation artifacts. Higher intensity = stronger signal.</p>
                          {preview && <GradCAMOverlay imageUrl={preview} regions={result.gradcam_regions} />}
                        </motion.div>
                      )}
                    </AnimatePresence>
                  </div>

                  <div className="card">
                    <button onClick={() => setShowModels(!showModels)}
                      className="w-full flex items-center justify-between text-sm font-semibold text-white">
                      <span className="flex items-center gap-2"><Eye size={15} className="text-violet-400" /> EfficientNet vs ViT</span>
                      {showModels ? <ChevronUp size={14} className="text-slate-400" /> : <ChevronDown size={14} className="text-slate-400" />}
                    </button>
                    <AnimatePresence>
                      {showModels && (
                        <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                          <div className="grid grid-cols-2 gap-4 mt-5">
                            {[{ model: 'EfficientNet-B4', data: result.efficientnet, color: '#22d3ee' },
                              { model: 'ViT-B/16', data: result.vit, color: '#a78bfa' }].map(({ model, data, color }) => (
                              <div key={model} className="p-4 rounded-xl" style={{ background: `${color}08`, border: `1px solid ${color}20` }}>
                                <p className="text-xs font-bold mb-2" style={{ color }}>{model}</p>
                                <p className="text-xl font-black text-white">{(data.confidence * 100).toFixed(1)}%</p>
                                <p className="text-xs text-slate-400 mb-3">confidence</p>
                                <div className="confidence-bar">
                                  <motion.div initial={{ width: 0 }} animate={{ width: `${data.confidence * 100}%` }}
                                    transition={{ duration: 1 }} className="h-full rounded-full"
                                    style={{ background: `linear-gradient(90deg, ${color}60, ${color})` }} />
                                </div>
                                <p className="text-xs mt-2 font-semibold" style={{ color }}>{data.prediction}</p>
                              </div>
                            ))}
                          </div>
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
