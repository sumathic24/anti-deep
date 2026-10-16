import { useState, useRef, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Upload, Image as ImageIcon, Loader2, ScanSearch, Eye, Brain, Info, X } from 'lucide-react'
import { analyzeImage, getErrorMessage } from '../utils/api'
import {
  PageHeader, Alert, VerdictCard, CollapsibleCard, EmptyState, LoadingState, ModelComparison, fadeIn,
} from '../components/ui'

// Must match ALLOWED_FORMATS in backend/routers/image.py
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** The image at its natural aspect ratio with GradCAM regions overlaid (coordinates are % of the image). */
function GradCAMOverlay({ imageUrl, regions }) {
  return (
    <div className="flex justify-center rounded-lg border border-line bg-canvas p-2">
      <div className="relative w-fit">
        <img src={imageUrl} alt="Analyzed image with GradCAM regions" className="block max-h-[28rem] w-auto max-w-full rounded" />
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            {regions.map((r, i) => (
              <radialGradient key={i} id={`rg${i}`} cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor="#e66767" stopOpacity={r.intensity * 0.85} />
                <stop offset="55%" stopColor="#e66767" stopOpacity={r.intensity * 0.5} />
                <stop offset="100%" stopColor="#e66767" stopOpacity="0" />
              </radialGradient>
            ))}
          </defs>
          {regions.map((r, i) => (
            <ellipse key={i} cx={r.x + r.w / 2} cy={r.y + r.h / 2} rx={r.w / 2} ry={r.h / 2} fill={`url(#rg${i})`} />
          ))}
        </svg>
        {regions.map((r, i) => (
          <span key={i} className="absolute rounded bg-black/75 px-1.5 py-0.5 text-[10px] font-medium text-white"
            style={{ left: `${r.x}%`, top: `${r.y}%` }}>
            {Math.round(r.intensity * 100)}%
          </span>
        ))}
      </div>
    </div>
  )
}

function Results({ result, preview }) {
  const hasModels = Boolean(result.efficientnet)
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show" className="flex flex-col gap-4">
      <VerdictCard result={result} otherLabel="Real" />

      {result.signals?.length > 0 && (
        <div className="card p-5 sm:p-6">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">Heuristic signals found</p>
          <ul className="space-y-2 text-sm text-fg-2">
            {result.signals.map((s) => (
              <li key={s} className="flex items-start gap-2">
                <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-fg-3" /> {s}
              </li>
            ))}
          </ul>
        </div>
      )}

      {!hasModels && (
        <Alert tone="info">
          GradCAM heatmaps and the EfficientNet vs ViT comparison need the trained image models.
          This result comes from <span className="text-fg">{result.model_used}</span>.
        </Alert>
      )}

      {hasModels && (
        <CollapsibleCard title="GradCAM heatmap" icon={Eye}
          description="Highlighted regions influenced EfficientNet's AI-generated prediction the most. Labels show the activation strength.">
          {result.gradcam_regions.length > 0 ? (
            preview && <GradCAMOverlay imageUrl={preview} regions={result.gradcam_regions} />
          ) : (
            <p className="text-sm text-fg-3">No regions highlighted — EfficientNet classified this image as real.</p>
          )}
        </CollapsibleCard>
      )}

      {hasModels && result.vit && (
        <CollapsibleCard title="Model comparison" icon={Brain} description="Final likelihood is the average of both models.">
          <ModelComparison models={[
            { name: 'EfficientNet-B4', ai: result.efficientnet.confidence, prediction: result.efficientnet.prediction },
            { name: 'ViT-B/16', ai: result.vit.confidence, prediction: result.vit.prediction },
          ]} />
        </CollapsibleCard>
      )}
    </motion.div>
  )
}

export default function ImageAnalysis() {
  const [file, setFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError('Unsupported file type. Please upload a JPG, PNG, WEBP or GIF image.')
      return
    }
    setFile(f); setResult(null); setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  const clearFile = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const handleAnalyze = async () => {
    if (!file || loading) return
    setLoading(true); setResult(null); setError(null)
    try {
      setResult(await analyzeImage(file))
    } catch (err) {
      setError(getErrorMessage(err))
    } finally { setLoading(false) }
  }

  const openPicker = () => inputRef.current?.click()

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="Computer vision" title="Image analysis"
        description="Check whether an image is AI-generated. With the trained models, GradCAM heatmaps show which regions drove the verdict." />

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        {/* Input */}
        <div className="flex flex-col gap-4 lg:col-span-5">
          <div className="card p-5 sm:p-6">
            <p className="mb-3 text-sm font-semibold text-fg">Image</p>
            <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(',')} className="hidden"
              onChange={(e) => { handleFile(e.target.files[0]); e.target.value = '' }} />

            {preview ? (
              <div className="overflow-hidden rounded-lg border border-line bg-canvas">
                <img src={preview} alt="Selected image preview" className="mx-auto block max-h-72 w-auto max-w-full" />
              </div>
            ) : (
              <div role="button" tabIndex={0} onClick={openPicker}
                onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker() } }}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true) }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) }}
                className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-14 text-center transition-colors ${
                  dragOver ? 'border-accent bg-accent/5' : 'border-line-strong hover:border-fg-3 hover:bg-white/[0.02]'}`}>
                <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg border border-line bg-raised">
                  <Upload size={18} className="text-accent" />
                </div>
                <p className="text-sm font-medium text-fg">Drop an image here or click to browse</p>
                <p className="mt-1 text-xs text-fg-3">JPG, PNG, WEBP or GIF</p>
              </div>
            )}

            {file && (
              <div className="mt-3 flex items-center gap-3 rounded-lg border border-line bg-raised px-3 py-2.5 text-sm">
                <ImageIcon size={16} className="shrink-0 text-fg-3" />
                <span className="min-w-0 flex-1 truncate text-fg">{file.name}</span>
                <span className="tabular shrink-0 text-xs text-fg-3">{formatSize(file.size)}</span>
                <button type="button" onClick={clearFile} className="btn-ghost -mr-1 p-1" aria-label="Remove image">
                  <X size={15} />
                </button>
              </div>
            )}

            {error && <div className="mt-4"><Alert>{error}</Alert></div>}

            <div className="mt-5 flex gap-2 border-t border-line pt-4">
              {file && (
                <button type="button" className="btn-secondary" onClick={openPicker} disabled={loading}>Change image</button>
              )}
              <button id="analyze-image-btn" type="button" onClick={handleAnalyze} disabled={!file || loading}
                className="btn-primary ml-auto">
                {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><ScanSearch size={16} /> Analyze image</>}
              </button>
            </div>
          </div>

          <div className="card p-5 sm:p-6">
            <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-fg">
              <Info size={15} className="text-fg-3" /> How detection works
            </p>
            <ul className="space-y-2 text-[13px] leading-relaxed text-fg-2">
              <li><span className="text-fg">Pre-trained detector (recommended):</span> a Community Forensics ViT trained on 4,800+ image generators, once downloaded with <code className="font-mono text-xs text-fg">download_models.py</code>.</li>
              <li><span className="text-fg">Trained models:</span> the project's own EfficientNet-B4 + ViT-B/16, averaged.</li>
              <li><span className="text-fg">Fallback:</span> a metadata heuristic (camera EXIF data) when neither is available.</li>
              <li><span className="text-fg">Explanation:</span> GradCAM highlights the regions behind an AI-generated verdict.</li>
            </ul>
          </div>
        </div>

        {/* Results */}
        <div className="lg:col-span-7" aria-live="polite">
          {loading ? (
            <LoadingState title="Analyzing image" description="Running the vision models…" />
          ) : result ? (
            <Results result={result} preview={preview} />
          ) : (
            <EmptyState icon={ImageIcon} title="No analysis yet"
              description="Upload an image, then select Analyze image. The verdict and its explanation will appear here." />
          )}
        </div>
      </div>
    </main>
  )
}
