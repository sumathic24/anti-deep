import { useRef, useState } from 'react'
import { Loader2, ScanSearch, Upload, Video as VideoIcon, X } from 'lucide-react'
import { analyzeVideo, getErrorMessage, isCancel } from '../utils/api'
import { ApiOfflineBanner, EmptyState, LoadingState, PageHeader, Alert, VerdictCard } from '../components/ui'

const ACCEPTED_TYPES = ['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/webm', 'video/x-matroska']
const MAX_MB = 200

export default function VideoAnalysis() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)
  const [progress, setProgress] = useState(null)
  const inputRef = useRef(null)
  const abortRef = useRef(null)

  const handleFile = (selected) => {
    if (!selected) return
    if (!ACCEPTED_TYPES.includes(selected.type) && !/\.(mp4|mov|avi|mkv|webm)$/i.test(selected.name)) {
      setError('Unsupported video type. Please upload MP4, MOV, AVI, MKV or WEBM.')
      return
    }
    if (selected.size > MAX_MB * 1024 * 1024) {
      setError(`This video is ${(selected.size / 1024 / 1024).toFixed(1)} MB. The limit is ${MAX_MB} MB.`)
      return
    }
    setFile(selected); setResult(null); setError(null)
  }

  const analyze = async () => {
    if (!file || loading) return
    const controller = new AbortController()
    abortRef.current = controller
    setLoading(true); setError(null); setProgress(0)
    try {
      setResult(await analyzeVideo(file, { signal: controller.signal, onUploadProgress: setProgress }))
    } catch (err) {
      if (!isCancel(err)) setError(getErrorMessage(err))
    } finally {
      setLoading(false); setProgress(null)
    }
  }

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="Video forensics" title="Video analysis"
        description="Sample frames from a video and classify it with the Hugging Face ConvNeXt deepfake detector." />
      <ApiOfflineBanner />
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
        <div className="card p-5 sm:p-6 lg:col-span-5">
          <p className="mb-3 text-sm font-semibold text-fg">Video</p>
          <input ref={inputRef} type="file" accept={ACCEPTED_TYPES.join(',')} className="hidden"
            onChange={(e) => { handleFile(e.target.files[0]); e.target.value = '' }} />
          <button type="button" onClick={() => inputRef.current?.click()}
            className="flex w-full cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-line-strong px-6 py-14 text-center hover:border-fg-3">
            <Upload size={18} className="mb-3 text-accent" />
            <p className="text-sm font-medium text-fg">Choose a video</p>
            <p className="mt-1 text-xs text-fg-3">MP4, MOV, AVI, MKV or WEBM · up to {MAX_MB} MB</p>
          </button>
          {file && <div className="mt-3 flex items-center gap-3 rounded-lg border border-line bg-raised px-3 py-2.5 text-sm">
            <VideoIcon size={16} className="shrink-0 text-fg-3" /><span className="min-w-0 flex-1 truncate">{file.name}</span>
            <button type="button" onClick={() => { setFile(null); setResult(null) }} className="btn-ghost p-1" aria-label="Remove video"><X size={15} /></button>
          </div>}
          {error && <div className="mt-4"><Alert>{error}</Alert></div>}
          <button type="button" onClick={analyze} disabled={!file || loading} className="btn-primary mt-5 w-full">
            {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing…</> : <><ScanSearch size={16} /> Analyze video</>}
          </button>
        </div>
        <div className="lg:col-span-7">
          {loading ? <LoadingState title="Analyzing video" description="Sampling frames and running the video detector…" progress={progress} onCancel={() => abortRef.current?.abort()} />
            : result ? <VerdictCard result={result} otherLabel="Real video" />
              : <EmptyState icon={VideoIcon} title="No analysis yet" description="Choose a video, then select Analyze video." />}
        </div>
      </div>
    </main>
  )
}
