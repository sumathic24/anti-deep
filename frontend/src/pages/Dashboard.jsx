import { useState, useEffect, useCallback } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import { FileText, Image as ImageIcon, Activity, RefreshCw, Loader2, AlertTriangle, CheckCircle2, Gauge } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { getDashboardStats, getErrorMessage } from '../utils/api'
import { PageHeader, Alert, EmptyState, VerdictBadge, VERDICT_COLORS, fadeIn } from '../components/ui'

const SURFACE = '#0d1320'
const SERIES = [
  { key: 'human', label: 'Human / real', color: VERDICT_COLORS.human.mark },
  { key: 'ai', label: 'AI-generated', color: VERDICT_COLORS.ai.mark },
]
const BREAKDOWN = [
  { key: 'ai_text', label: 'AI-generated text', kind: 'text', isAI: true },
  { key: 'human_text', label: 'Human-written text', kind: 'text', isAI: false },
  { key: 'ai_image', label: 'AI-generated image', kind: 'image', isAI: true },
  { key: 'real_image', label: 'Real image', kind: 'image', isAI: false },
]

const formatDate = (date) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(undefined, { weekday: 'short', day: 'numeric', month: 'short' })
const formatDateTime = (createdAt) =>
  new Date(createdAt.replace(' ', 'T')).toLocaleString(undefined, { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })

/** Stacked segment: 4px rounded data-end on the top-most non-empty segment only. */
function Segment({ x, y, width, height, fill, payload, seriesKey }) {
  if (!height || height <= 0) return null
  const isTop = seriesKey === 'ai' || !payload.ai
  const r = isTop ? Math.min(4, height, width / 2) : 0
  const d = `M${x},${y + height} V${y + r} Q${x},${y} ${x + r},${y} H${x + width - r} Q${x + width},${y} ${x + width},${y + r} V${y + height} Z`
  // The surface-colored stroke leaves a 2px gap between touching segments.
  return <path d={d} fill={fill} stroke={SURFACE} strokeWidth={2} />
}

function WeeklyTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-lg border border-line-strong bg-raised px-3 py-2.5 text-xs shadow-xl">
      <p className="mb-2 text-fg-3">{formatDate(d.date)}</p>
      {[...SERIES].reverse().map((s) => (
        <p key={s.key} className="flex items-center gap-2 py-0.5">
          <span className="h-0.5 w-3 rounded-full" style={{ background: s.color }} />
          <span className="tabular w-5 font-semibold text-fg">{d[s.key]}</span>
          <span className="text-fg-3">{s.label}</span>
        </p>
      ))}
    </div>
  )
}

function StatTile({ label, value, sub, icon: Icon }) {
  return (
    <div className="card p-5">
      <div className="flex items-center justify-between">
        <p className="text-sm text-fg-2">{label}</p>
        <Icon size={16} className="text-fg-3" />
      </div>
      <p className="mt-3 text-3xl font-semibold tracking-tight text-fg">{value}</p>
      {sub && <p className="mt-1 text-xs text-fg-3">{sub}</p>}
    </div>
  )
}

function WeeklyChart({ weekly }) {
  const [view, setView] = useState('chart')
  return (
    <div className="card p-5 sm:p-6 lg:col-span-2">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold text-fg">Verdicts in the last 7 days</h2>
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-fg-2">
            {[...SERIES].reverse().map((s) => (
              <span key={s.key} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} /> {s.label}
              </span>
            ))}
          </div>
        </div>
        <div className="flex rounded-lg border border-line-strong p-0.5 text-xs" role="group" aria-label="View">
          {['chart', 'table'].map((v) => (
            <button key={v} type="button" onClick={() => setView(v)} aria-pressed={view === v}
              className={`rounded-md px-2.5 py-1 capitalize transition-colors ${view === v ? 'bg-white/10 text-fg' : 'text-fg-3 hover:text-fg'}`}>
              {v}
            </button>
          ))}
        </div>
      </div>
      {view === 'chart' ? (
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={weekly} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke="#1c2536" />
            <XAxis dataKey="day" tick={{ fill: '#7c869a', fontSize: 11 }} axisLine={{ stroke: '#2a3549' }} tickLine={false} />
            <YAxis allowDecimals={false} tick={{ fill: '#7c869a', fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
            <Tooltip content={<WeeklyTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
            {SERIES.map((s) => (
              <Bar key={s.key} dataKey={s.key} stackId="verdicts" fill={s.color} maxBarSize={24} isAnimationActive={false}
                shape={(props) => <Segment {...props} seriesKey={s.key} />} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="overflow-x-auto">
          <table className="table-base">
            <thead><tr><th>Day</th><th className="text-right">AI-generated</th><th className="text-right">Human / real</th><th className="text-right">Total</th></tr></thead>
            <tbody>
              {weekly.map((d) => (
                <tr key={d.date}>
                  <td>{formatDate(d.date)}</td>
                  <td className="tabular text-right">{d.ai}</td>
                  <td className="tabular text-right">{d.human}</td>
                  <td className="tabular text-right font-medium text-fg">{d.ai + d.human}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}

function Breakdown({ stats }) {
  return (
    <div className="card p-5 sm:p-6">
      <h2 className="font-semibold text-fg">Breakdown</h2>
      <p className="mb-5 mt-1 text-xs text-fg-3">All analyses by verdict, with the model's average confidence</p>
      <ul className="space-y-4">
        {BREAKDOWN.map((b) => {
          const count = stats.breakdown[b.key]
          const share = stats.total_analyses ? count / stats.total_analyses : 0
          const conf = stats.avg_confidence_by_type[b.key]
          return (
            <li key={b.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-2 text-sm">
                <span className="text-fg-2">{b.label}</span>
                <span className="tabular text-fg">{count} <span className="text-xs text-fg-3">({Math.round(share * 100)}%)</span></span>
              </div>
              <div className="meter h-1.5">
                <div className="h-full rounded-full bg-fg-3" style={{ width: `${share * 100}%` }} />
              </div>
              <p className="mt-1 text-xs text-fg-3">{conf == null ? 'No results yet' : `Avg confidence ${conf}%`}</p>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function RecentTable({ recent }) {
  return (
    <div className="card overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 sm:px-6">
        <h2 className="font-semibold text-fg">Recent analyses</h2>
        <Link to="/text-analysis" className="text-sm font-medium text-accent hover:underline">New analysis</Link>
      </div>
      {/* Phones: stacked list */}
      <ul className="divide-y divide-line border-t border-line sm:hidden">
        {recent.map((a, i) => (
          <li key={i} className="px-5 py-4">
            <div className="mb-2 flex items-center justify-between gap-3">
              <VerdictBadge isAI={a.is_ai}>{a.prediction}</VerdictBadge>
              <span className="tabular text-sm text-fg">{a.confidence}%</span>
            </div>
            <p className="truncate text-sm text-fg" title={a.label}>{a.kind === 'text' ? `“${a.label}”` : a.label}</p>
            <p className="mt-1 flex items-center gap-1.5 text-xs text-fg-3">
              {a.kind === 'text' ? <FileText size={12} /> : <ImageIcon size={12} />}
              {a.kind === 'text' ? 'Text' : 'Image'} · {formatDateTime(a.created_at)}
            </p>
          </li>
        ))}
      </ul>
      {/* Larger screens: table */}
      <div className="hidden overflow-x-auto border-t border-line sm:block">
        <table className="table-base min-w-[40rem]">
          <thead>
            <tr><th>Type</th><th>Input</th><th>Verdict</th><th className="text-right">Confidence</th><th className="text-right">When</th></tr>
          </thead>
          <tbody>
            {recent.map((a, i) => (
              <tr key={i}>
                <td>
                  <span className="flex items-center gap-2 text-fg-2">
                    {a.kind === 'text' ? <FileText size={14} /> : <ImageIcon size={14} />}
                    {a.kind === 'text' ? 'Text' : 'Image'}
                  </span>
                </td>
                <td className="max-w-[18rem] truncate text-fg" title={a.label}>{a.kind === 'text' ? `“${a.label}”` : a.label}</td>
                <td><VerdictBadge isAI={a.is_ai}>{a.prediction}</VerdictBadge></td>
                <td className="tabular text-right">{a.confidence}%</td>
                <td className="tabular whitespace-nowrap text-right text-fg-3">{formatDateTime(a.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [stats, setStats] = useState(null)
  const [error, setError] = useState(null)
  const [refreshing, setRefreshing] = useState(false)

  const load = useCallback(() => {
    setRefreshing(true)
    return getDashboardStats()
      .then((data) => { setStats(data); setError(null) })
      .catch((err) => setError(getErrorMessage(err)))
      .finally(() => setRefreshing(false))
  }, [])

  useEffect(() => { load() }, [load])

  const hasData = stats && stats.total_analyses > 0

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="Analytics" title="Dashboard"
        description="Every text and image analysis run on this server is recorded here."
        actions={
          <button type="button" className="btn-secondary" onClick={load} disabled={refreshing}>
            <RefreshCw size={15} className={refreshing ? 'animate-spin' : ''} /> Refresh
          </button>
        } />

      {error && <div className="mb-6"><Alert>{error}</Alert></div>}

      {!stats && !error && (
        <div className="card flex items-center justify-center gap-3 py-24 text-sm text-fg-3" role="status">
          <Loader2 size={18} className="animate-spin text-accent" /> Loading statistics…
        </div>
      )}

      {stats && !hasData && (
        <EmptyState icon={Activity} title="No analyses yet"
          description="Run a text or image analysis and the results will show up here.">
          <div className="flex flex-wrap justify-center gap-3">
            <Link to="/text-analysis" className="btn-primary"><FileText size={15} /> Analyze text</Link>
            <Link to="/image-analysis" className="btn-secondary"><ImageIcon size={15} /> Analyze an image</Link>
          </div>
        </EmptyState>
      )}

      {hasData && (
        <motion.div variants={fadeIn} initial="hidden" animate="show"
          className={`flex flex-col gap-6 transition-opacity ${refreshing ? 'opacity-60' : ''}`}>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile label="Total analyses" value={stats.total_analyses.toLocaleString()} icon={Activity}
              sub={`${stats.text_analyses} text · ${stats.image_analyses} image`} />
            <StatTile label="AI-generated" value={stats.ai_content_detected.toLocaleString()} icon={AlertTriangle}
              sub={`${Math.round(stats.ai_content_detected / stats.total_analyses * 100)}% of all analyses`} />
            <StatTile label="Human / real" value={stats.verified_real.toLocaleString()} icon={CheckCircle2}
              sub={`${Math.round(stats.verified_real / stats.total_analyses * 100)}% of all analyses`} />
            <StatTile label="Average confidence" value={`${stats.avg_confidence}%`} icon={Gauge}
              sub="In the predicted verdict" />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <WeeklyChart weekly={stats.weekly} />
            <Breakdown stats={stats} />
          </div>

          <RecentTable recent={stats.recent} />
        </motion.div>
      )}
    </main>
  )
}
