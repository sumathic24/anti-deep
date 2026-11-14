import { useState } from 'react'
import { Film, Loader2, Upload, X } from 'lucide-react'
import { analyzeVideo } from '../utils/api'

const MAX_UPLOAD_MB = 200
const ACCEPTED_EXTENSIONS = /\.(mp4|mov|avi|mkv|webm)$/i

export default function VideoAnalysis() {
  const [file, setFile] = useState(null)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [loading, setLoading] = useState(false)

  const selectFile = (selected) => {
    if (!selected) return
    if (!ACCEPTED_EXTENSIONS.test(selected.name)) {
      setError('Unsupported video format. Use MP4, MOV, AVI, MKV, or WEBM.')
      return
    }
    if (selected.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setError(`The video must be smaller than ${MAX_UPLOAD_MB} MB.`)
      return
    }
    setFile(selected)
    setResult(null)
    setError(null)
  }

  const handleAnalyze = async () => {
    if (!file || loading) return
    setLoading(true)
    setError(null)
    try {
      setResult(await analyzeVideo(file))
    } catch (requestError) {
      setError(requestError.response?.data?.detail || 'Video analysis is unavailable.')
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        <p className="section-label mb-2">Video Forensics</p>
        <h1 className="text-4xl font-bold text-white mb-2">Video <span className="gradient-text">Analysis</span></h1>
        <p className="text-slate-400 max-w-2xl mb-10">
          Detect deepfake content directly from your video with a native video classifier.
        </p>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="card">
            <label className="flex min-h-56 cursor-pointer flex-col items-center justify-center rounded-xl border border-dashed border-white/15 text-center">
              <input
                type="file"
                accept="video/mp4,video/quicktime,video/x-msvideo,video/webm,video/x-matroska"
                className="hidden"
                onChange={(event) => {
                  selectFile(event.target.files?.[0])
                  event.target.value = ''
                }}
              />
              <Upload size={24} className="text-cyan-400 mb-3" />
              <span className="text-white font-semibold">Choose a video</span>
              <span className="text-slate-500 text-xs mt-2">MP4, MOV, AVI, MKV, or WEBM up to 200 MB</span>
            </label>

            {file && (
              <div className="mt-4 flex items-center gap-3 rounded-xl bg-white/[0.03] p-3">
                <Film size={18} className="text-cyan-400" />
                <span className="text-sm text-slate-300 truncate flex-1">{file.name}</span>
                <button type="button" onClick={() => { setFile(null); setResult(null) }} className="text-slate-500 hover:text-white" aria-label="Remove video">
                  <X size={17} />
                </button>
              </div>
            )}

            {error && <p className="mt-4 text-sm text-rose-400">{error}</p>}
            <button type="button" onClick={handleAnalyze} disabled={!file || loading} className="btn-primary w-full mt-5 flex items-center justify-center gap-2 disabled:opacity-50">
              {loading ? <><Loader2 size={16} className="animate-spin" /> Analyzing...</> : 'Analyze video'}
            </button>
          </div>

          <div className="card">
            {!result ? (
              <div className="min-h-56 flex flex-col items-center justify-center text-center">
                <Film size={30} className="text-slate-600 mb-3" />
                <p className="text-slate-400">Your video result will appear here.</p>
              </div>
            ) : (
              <>
                <p className="section-label mb-3">Detection result</p>
                <h2 className={`text-3xl font-black ${result.ai_confidence > 0.5 ? 'text-rose-400' : 'text-cyan-400'}`}>
                  {result.prediction}
                </h2>
                <p className="text-slate-400 mt-3">AI confidence: {(result.ai_confidence * 100).toFixed(1)}%</p>
                <div className="confidence-bar mt-4">
                  <div className="h-full rounded-full bg-gradient-to-r from-cyan-400 to-rose-400" style={{ width: `${result.ai_confidence * 100}%` }} />
                </div>
                <p className="text-xs text-slate-500 mt-4">{result.model_used}</p>
                <div className="mt-6 border-t border-white/10 pt-5">
                  <p className="text-sm font-semibold text-white">Native video classifier</p>
                  <p className="text-xs text-slate-500 mt-2">
                  The uploaded video is classified directly on the CPU without frame extraction.
                  </p>
                  {result.gradcam_regions?.length ? (
                    <div className="grid grid-cols-2 gap-2 mt-4">
                      {result.gradcam_regions.map((region, index) => (
                        <div key={`${region.x}-${region.y}-${index}`} className="rounded-lg bg-rose-400/10 border border-rose-400/20 px-3 py-2">
                          <p className="text-xs text-rose-300">{region.label}</p>
                          <p className="text-xs text-slate-400 mt-1">Intensity: {(region.intensity * 100).toFixed(1)}%</p>
                        </div>
                      ))}
                    </div>
                  ) : (
                    <p className="text-xs text-slate-500 mt-4">No high-signal regions were found.</p>
                  )}
                </div>
              </>
            )}
          </div>
        </div>
      </div>
    </main>
  )
}
