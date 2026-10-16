import { useState, useRef, useCallback, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Upload, Image as ImageIcon, Loader2, ScanSearch, Eye, Brain, Info, X } from 'lucide-react'
import { analyzeImage, getErrorMessage, isCancel } from '../utils/api'
import { useTheme } from '../utils/theme'
import { usePersistentState } from '../utils/usePersistentState'
import {
  PageHeader, Alert, ApiOfflineBanner, VerdictCard, ReportActions, ReportHeader, CollapsibleCard,
  EmptyState, LoadingState, ModelComparison, fadeIn, scrollToOnMobile,
} from '../components/ui'

// Must match ALLOWED_FORMATS and MAX_UPLOAD_MB in backend/routers/image.py
const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/webp', 'image/gif']
const MAX_MB = 20

function formatSize(bytes) {
  return bytes < 1024 * 1024 ? `${Math.max(1, Math.round(bytes / 1024))} KB` : `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/** The image at its natural aspect ratio with GradCAM regions overlaid (coordinates are % of the image). */
function GradCAMOverlay({ imageUrl, regions }) {
  const { colors } = useTheme()
  return (
    <div className="flex justify-center rounded-lg border border-line bg-canvas p-2">
      <div className="relative w-fit">
        <img src={imageUrl} alt="Analyzed image with GradCAM regions" className="block max-h-[28rem] w-auto max-w-full rounded" />
        <svg className="absolute inset-0 h-full w-full" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <defs>
            {regions.map((r, i) => (
              <radialGradient key={i} id={`rg${i}`} cx="50%" cy="50%" r="50%">
                <stop offset="0%" stopColor={colors.ai} stopOpacity={r.intensity * 0.85} />
                <stop offset="55%" stopColor={colors.ai} stopOpacity={r.intensity * 0.5} />
                <stop offset="100%" stopColor={colors.ai} stopOpacity="0" />
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

function buildSummary(fileName, r) {
  const confidence = (Math.max(r.ai_confidence, 1 - r.ai_confidence) * 100).toFixed(1)
  return [
    `DeepFakeShield — image analysis (${new Date().toLocaleString()})`,
    `Image: ${fileName}`,
    `Verdict: ${r.prediction} (${confidence}% confidence)`,
    `AI likelihood: ${(r.ai_confidence * 100).toFixed(1)}%`,
    `Model: ${r.model_used}`,
    ...(r.signals?.length ? [`Signals: ${r.signals.join('; ')}`] : []),
  ].join('\n')
}

/** Shown only when printing (Print / save as PDF). */
function PrintReport({ fileName, preview, result }) {
  return (
    <div className="hidden print:block">
      <ReportHeader title="Image analysis report" />
      <VerdictCard result={result} otherLabel="Real" />
      <div className="card mt-4 p-5">
        <p className="mb-3 text-sm font-semibold">Analyzed image — {fileName}</p>
        {preview && <img src={preview} alt="" className="max-h-96 w-auto max-w-full" />}
        {result.signals?.length > 0 && <p className="mt-3 text-sm text-fg-2">Signals: {result.signals.join('; ')}</p>}
      </div>
    </div>
  )
}

function Results({ result, preview, fileName }) {
  const hasModels = Boolean(result.efficientnet)
  return (
    <motion.div variants={fadeIn} initial="hidden" animate="show" className="flex flex-col gap-4">
      <VerdictCard result={result} otherLabel="Real"
        actions={<ReportActions summary={buildSummary(fileName, result)} />} />

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
          GradCAM heatmaps and the EfficientNet vs ViT comparison need the project's own trained image models.
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
  const [file, setFile] = usePersistentState('image.file', null)
  const [preview, setPreview] = usePersistentState('image.preview', null)
  const [result, setResult] = usePersistentState('image.result', null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [error, setError] = useState(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)
  const resultsRef = useRef(null)

  const handleFile = useCallback((f) => {
    if (!f) return
    if (!ACCEPTED_TYPES.includes(f.type)) {
      setError('Unsupported file type. Please upload a JPG, PNG, WEBP or GIF image.')
      return
    }
    if (f.size > MAX_MB * 1024 * 1024) {
      setError(`This image is ${formatSize(f.size)}. The limit is ${MAX_MB} MB.`)
      return
    }
    setFile(f); setResult(null); setError(null)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [setFile, setResult, setPreview])

  // Paste an image from the clipboard (Ctrl+V) anywhere on the page.
  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
      const blob = item?.getAsFile()
      if (!blob) return
      e.preventDefault()
      const ext = blob.type.split('/')[1] || 'png'
      handleFile(new File([blob], blob.name && blob.name !== 'image.png' ? blob.name : `pasted-image.${ext}`, { type: blob.type }))
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  }, [handleFile])

  useEffect(() => () => abortRef.current?.abort(), [])

  const clearFile = () => { setFile(null); setPreview(null); setResult(null); setError(null) }

  const handleAnalyze = async () => {
    if (!file || loading) return
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true); setResult(null); setError(null); setProgress(0)
    requestAnimationFrame(() => scrollToOnMobile(resultsRef))
    try {
      setResult(await analyzeImage(file, { signal: controller.signal, onUploadProgress: setProgress }))
      requestAnimationFrame(() => scrollToOnMobile(resultsRef))
    } catch (err) {
      if (!isCancel(err)) setError(getErrorMessage(err))
    } finally { setLoading(false); setProgress(null) }
  }

  const openPicker = () => inputRef.current?.click()
  const dropHandlers = {
    onDragOver: (e) => { e.preventDefault(); setDragOver(true) },
    onDragLeave: () => setDragOver(false),
    onDrop: (e) => { e.preventDefault(); setDragOver(false); handleFile(e.dataTransfer.files[0]) },
  }

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="Computer vision" title="Image analysis"
        description="Check whether an image is AI-generated. With the project's own trained models, GradCAM heatmaps show which regions drove the verdict." />
      <ApiOfflineBanner />

      {/* Phones: input → results → info.  Desktop: input and info on the left, results on the right. */}
      <div className="grid grid-cols-1 gap-6 print:hidden lg:grid-cols-12 lg:grid-rows-[auto_1fr] lg:items-start">
        <div className="card p-5 sm:p-6 lg:col-span-5">
          <p className="mb-3 text-sm font-semibold text-fg">Image</p>
          <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(',')} className="hidden"
            onChange={(e) => { handleFile(e.target.files[0]); e.target.value = '' }} />

          {preview ? (
            <div {...dropHandlers} className="relative overflow-hidden rounded-lg border border-line bg-canvas">
              <img src={preview} alt="Selected image preview" className="mx-auto block max-h-72 w-auto max-w-full" />
              {dragOver && (
                <div className="absolute inset-0 flex items-center justify-center bg-canvas/85 text-sm font-medium text-fg">
                  Drop to replace the image
                </div>
              )}
            </div>
          ) : (
            <div role="button" tabIndex={0} onClick={openPicker} {...dropHandlers}
              onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openPicker() } }}
              className={`flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed px-6 py-14 text-center transition-colors ${
                dragOver ? 'border-accent bg-accent/5' : 'border-line-strong hover:border-fg-3 hover:bg-fg/[0.03]'}`}>
              <div className="mb-3 flex h-11 w-11 items-center justify-center rounded-lg border border-line bg-raised">
                <Upload size={18} className="text-accent" />
              </div>
              <p className="text-sm font-medium text-fg">Drop an image here or click to browse</p>
              <p className="mt-1 text-xs text-fg-3">JPG, PNG, WEBP or GIF · up to {MAX_MB} MB · or paste with Ctrl+V</p>
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
          {file && <p className="mt-2 text-xs text-fg-3">Drop or paste another image to replace this one.</p>}

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

        <div ref={resultsRef} className="scroll-mt-20 lg:col-span-7 lg:col-start-6 lg:row-span-2 lg:row-start-1" aria-live="polite">
          {loading ? (
            <LoadingState title="Analyzing image" description="Uploading the image and running the vision model…"
              progress={progress} onCancel={() => abortRef.current?.abort()} />
          ) : result ? (
            <Results result={result} preview={preview} fileName={file?.name || 'image'} />
          ) : (
            <EmptyState icon={ImageIcon} title="No analysis yet"
              description="Upload or paste an image, then select Analyze image. The verdict and its explanation will appear here." />
          )}
        </div>

        <div className="card p-5 sm:p-6 lg:col-span-5">
          <p className="mb-3 flex items-center gap-2 text-sm font-semibold text-fg">
            <Info size={15} className="text-fg-3" /> How detection works
          </p>
          <ul className="space-y-2 text-[13px] leading-relaxed text-fg-2">
            <li><span className="text-fg">Pre-trained detector (recommended):</span> a Community Forensics ViT trained on 4,800+ image generators, once downloaded with <code className="font-mono text-xs text-fg">download_models.py</code>.</li>
            <li><span className="text-fg">Trained models:</span> the project's own EfficientNet-B4 + ViT-B/16, averaged.</li>
            <li><span className="text-fg">Fallback:</span> a metadata heuristic (camera EXIF data) when neither is available.</li>
            <li><span className="text-fg">Explanation:</span> GradCAM highlights the regions behind an AI-generated verdict (project's own EfficientNet only).</li>
          </ul>
        </div>
      </div>

      {result && <PrintReport fileName={file?.name || 'image'} preview={preview} result={result} />}
    </main>
  )
}
