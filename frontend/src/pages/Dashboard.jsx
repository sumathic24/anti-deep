import { useState, useEffect, useCallback, useRef } from 'react'
import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  FileText, Image as ImageIcon, Activity, RefreshCw, Loader2, AlertTriangle, CheckCircle2, Gauge,
  ChevronDown, Trash2, ShieldCheck,
} from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { getDashboardStats, getHistory, clearHistory, getErrorMessage } from '../utils/api'
import { useTheme } from '../utils/theme'
import { usePersistentState } from '../utils/usePersistentState'
import { PageHeader, Alert, ApiOfflineBanner, EmptyState, VerdictBadge, fadeIn } from '../components/ui'

const REFRESH_MS = 30000
const PAGE_SIZE = 10
const RANGES = [{ days: 7, label: '7 days' }, { days: 30, label: '30 days' }]
const BREAKDOWN = [
  { key: 'ai_text', label: 'AI-generated text' },
  { key: 'human_text', label: 'Human-written text' },
  { key: 'ai_image', label: 'AI-generated image' },
  { key: 'real_image', label: 'Real image' },
]

const formatDate = (date, opts = { weekday: 'short', day: 'numeric', month: 'short' }) =>
  new Date(`${date}T00:00:00`).toLocaleDateString(undefined, opts)
const formatDateTime = (createdAt, opts = { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' }) =>
  new Date(createdAt.replace(' ', 'T')).toLocaleString(undefined, opts)
const pct = (part, whole) => (whole ? Math.round((part / whole) * 100) : 0)

function useSeries() {
  const { colors } = useTheme()
  return [
    { key: 'human', label: 'Human / real', color: colors.human },
    { key: 'ai', label: 'AI-generated', color: colors.ai },
  ]
}

/** Stacked segment: 4px rounded data-end on the top-most non-empty segment only. */
function Segment({ x, y, width, height, fill, payload, seriesKey, surface }) {
  if (!height || height <= 0) return null
  const isTop = seriesKey === 'ai' || !payload.ai
  const r = isTop ? Math.min(4, height, width / 2) : 0
  const d = `M${x},${y + height} V${y + r} Q${x},${y} ${x + r},${y} H${x + width - r} Q${x + width},${y} ${x + width},${y + r} V${y + height} Z`
  // The surface-colored stroke leaves a 2px gap between touching segments.
  return <path d={d} fill={fill} stroke={surface} strokeWidth={2} />
}

function DailyTooltip({ active, payload, series }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="rounded-lg border border-line-strong bg-raised px-3 py-2.5 text-xs shadow-xl">
      <p className="mb-2 text-fg-3">{formatDate(d.date)}</p>
      {[...series].reverse().map((s) => (
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

function DailyChart({ daily, days }) {
  const { colors } = useTheme()
  const series = useSeries()
  const [view, setView] = useState('chart')
  const tickFormatter = (date) => (days <= 7 ? formatDate(date, { weekday: 'short' }) : formatDate(date, { day: 'numeric', month: 'short' }))
  return (
    <div className="card p-5 sm:p-6 lg:col-span-2">
      <div className="mb-5 flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="font-semibold text-fg">Verdicts per day</h2>
          <div className="mt-2 flex flex-wrap gap-4 text-xs text-fg-2">
            {[...series].reverse().map((s) => (
              <span key={s.key} className="flex items-center gap-1.5">
                <span className="h-2.5 w-2.5 rounded-sm" style={{ background: s.color }} /> {s.label}
              </span>
            ))}
          </div>
        </div>
        <div className="segmented" role="group" aria-label="View">
          {['chart', 'table'].map((v) => (
            <button key={v} type="button" onClick={() => setView(v)} aria-pressed={view === v} className="capitalize">{v}</button>
          ))}
        </div>
      </div>
      {view === 'chart' ? (
        <ResponsiveContainer width="100%" height={240}>
          <BarChart data={daily} margin={{ top: 4, right: 4, bottom: 0, left: -12 }}>
            <CartesianGrid vertical={false} stroke={colors.grid} />
            <XAxis dataKey="date" tickFormatter={tickFormatter} interval="preserveStartEnd" minTickGap={12}
              tick={{ fill: colors.tick, fontSize: 11 }} axisLine={{ stroke: colors.axis }} tickLine={false} />
            <YAxis allowDecimals={false} tick={{ fill: colors.tick, fontSize: 11 }} axisLine={false} tickLine={false} width={40} />
            <Tooltip content={<DailyTooltip series={series} />} cursor={{ fill: `${colors.fg}0a` }} />
            {series.map((s) => (
              <Bar key={s.key} dataKey={s.key} stackId="verdicts" fill={s.color} maxBarSize={24} isAnimationActive={false}
                shape={(props) => <Segment {...props} seriesKey={s.key} surface={colors.surface} />} />
            ))}
          </BarChart>
        </ResponsiveContainer>
      ) : (
        <div className="max-h-[240px] overflow-auto">
          <table className="table-base">
            <thead><tr><th>Day</th><th className="text-right">AI-generated</th><th className="text-right">Human / real</th><th className="text-right">Total</th></tr></thead>
            <tbody>
              {[...daily].reverse().map((d) => (
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
      <p className="mb-5 mt-1 text-xs text-fg-3">Analyses in this period by verdict, with the model's average confidence</p>
      <ul className="space-y-4">
        {BREAKDOWN.map((b) => {
          const count = stats.breakdown[b.key]
          const conf = stats.avg_confidence_by_type[b.key]
          return (
            <li key={b.key}>
              <div className="mb-1.5 flex items-baseline justify-between gap-2 text-sm">
                <span className="text-fg-2">{b.label}</span>
                <span className="tabular text-fg">{count} <span className="text-xs text-fg-3">({pct(count, stats.total_analyses)}%)</span></span>
              </div>
              <div className="meter h-1.5">
                <div className="h-full rounded-full bg-fg-3" style={{ width: `${pct(count, stats.total_analyses)}%` }} />
              </div>
              <p className="mt-1 text-xs text-fg-3">{conf == null ? 'No results in this period' : `Avg confidence ${conf}%`}</p>
            </li>
          )
        })}
      </ul>
    </div>
  )
}

function HistoryDetails({ item }) {
  return (
    <dl className="grid grid-cols-1 gap-3 text-xs sm:grid-cols-4">
      <div><dt className="text-fg-3">Recorded</dt><dd className="mt-0.5 text-fg-2">{formatDateTime(item.created_at, { dateStyle: 'medium', timeStyle: 'medium' })}</dd></div>
      <div><dt className="text-fg-3">Type</dt><dd className="mt-0.5 text-fg-2">{item.kind === 'text' ? 'Text' : 'Image'}</dd></div>
      <div><dt className="text-fg-3">Confidence in verdict</dt><dd className="tabular mt-0.5 text-fg-2">{item.confidence}%</dd></div>
      <div><dt className="text-fg-3">Model</dt><dd className="mt-0.5 break-words text-fg-2">{item.model_used || 'Not recorded'}</dd></div>
      <div className="sm:col-span-4"><dt className="text-fg-3">{item.kind === 'text' ? 'Stored text snippet' : 'File name'}</dt><dd className="mt-0.5 break-words text-fg">{item.label}</dd></div>
    </dl>
  )
}

function HistoryList({ items, total, onLoadMore, loadingMore, onCleared }) {
  const [expanded, setExpanded] = useState(null)
  const [confirming, setConfirming] = useState(false)
  const [clearing, setClearing] = useState(false)
  const [error, setError] = useState(null)
  const toggle = (id) => setExpanded((cur) => (cur === id ? null : id))

  const doClear = async () => {
    setClearing(true); setError(null)
    try { await clearHistory(); setConfirming(false); onCleared() }
    catch (err) { setError(getErrorMessage(err)) }
    finally { setClearing(false) }
  }

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 px-5 py-4 sm:px-6">
        <div>
          <h2 className="font-semibold text-fg">Analysis history</h2>
          <p className="mt-0.5 text-xs text-fg-3">{total} recorded · select a row for details</p>
        </div>
        {!confirming && (
          <button type="button" className="btn-ghost btn-sm" onClick={() => setConfirming(true)}>
            <Trash2 size={14} /> Clear history
          </button>
        )}
      </div>

      {confirming && (
        <div className="mx-5 mb-4 flex flex-col gap-3 rounded-lg border border-ai/30 bg-ai/10 p-4 sm:mx-6 sm:flex-row sm:items-center sm:justify-between" role="alertdialog" aria-label="Confirm clearing history">
          <p className="text-sm text-fg">Delete all {total} recorded analyses? This can't be undone.</p>
          <div className="flex shrink-0 gap-2">
            <button type="button" className="btn-secondary btn-sm" onClick={() => setConfirming(false)} disabled={clearing}>Cancel</button>
            <button type="button" className="btn-danger btn-sm" onClick={doClear} disabled={clearing}>
              {clearing ? <Loader2 size={14} className="animate-spin" /> : <Trash2 size={14} />} Delete all
            </button>
          </div>
        </div>
      )}
      {error && <div className="mx-5 mb-4 sm:mx-6"><Alert>{error}</Alert></div>}

      <ul className="divide-y divide-line border-t border-line">
        {items.map((a) => (
          <li key={a.id}>
            <button type="button" onClick={() => toggle(a.id)} aria-expanded={expanded === a.id}
              className="grid w-full grid-cols-[1fr_auto] items-center gap-x-4 gap-y-1.5 px-5 py-3.5 text-left transition-colors hover:bg-fg/[0.03] sm:grid-cols-[5.5rem_minmax(0,1fr)_9rem_5rem_8.5rem_1rem] sm:px-6">
              <span className="hidden items-center gap-2 text-sm text-fg-2 sm:flex">
                {a.kind === 'text' ? <FileText size={14} /> : <ImageIcon size={14} />}
                {a.kind === 'text' ? 'Text' : 'Image'}
              </span>
              <span className="col-span-2 min-w-0 truncate text-sm text-fg sm:col-span-1">{a.kind === 'text' ? `“${a.label}”` : a.label}</span>
              <span><VerdictBadge isAI={a.is_ai}>{a.prediction}</VerdictBadge></span>
              <span className="tabular text-right text-sm text-fg-2">{a.confidence}%</span>
              <span className="tabular hidden text-right text-sm text-fg-3 sm:block">{formatDateTime(a.created_at)}</span>
              <ChevronDown size={15} className={`hidden text-fg-3 transition-transform sm:block ${expanded === a.id ? 'rotate-180' : ''}`} />
            </button>
            {expanded === a.id && (
              <div className="border-t border-line bg-raised/60 px-5 py-4 sm:px-6">
                <HistoryDetails item={a} />
              </div>
            )}
          </li>
        ))}
      </ul>

      <div className="flex flex-col items-center gap-3 border-t border-line px-5 py-4 text-center sm:flex-row sm:justify-between sm:px-6 sm:text-left">
        <p className="flex items-center gap-1.5 text-xs text-fg-3">
          <ShieldCheck size={13} /> Only the first 80 characters of each text, or the image's file name, are stored — on this server only.
        </p>
        {items.length < total && (
          <button type="button" className="btn-secondary btn-sm" onClick={onLoadMore} disabled={loadingMore}>
            {loadingMore && <Loader2 size={14} className="animate-spin" />} Show more ({total - items.length} left)
          </button>
        )}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const [days, setDays] = usePersistentState('dashboard.days', 7)
  const [stats, setStats] = useState(null)
  const [history, setHistory] = useState({ items: [], total: 0 })
  const [error, setError] = useState(null)
  const [refreshing, setRefreshing] = useState(false)
  const [loadingMore, setLoadingMore] = useState(false)
  const loadedCount = useRef(PAGE_SIZE)

  const load = useCallback(async () => {
    setRefreshing(true)
    try {
      const [s, h] = await Promise.all([getDashboardStats(days), getHistory(Math.max(loadedCount.current, PAGE_SIZE), 0)])
      setStats(s); setHistory(h); setError(null)
    } catch (err) {
      setError(getErrorMessage(err))
    } finally { setRefreshing(false) }
  }, [days])

  useEffect(() => { load() }, [load])

  // Keep the numbers current while the page is open and visible.
  useEffect(() => {
    const timer = setInterval(() => { if (document.visibilityState === 'visible') load() }, REFRESH_MS)
    return () => clearInterval(timer)
  }, [load])

  const loadMore = async () => {
    setLoadingMore(true)
    try {
      const next = await getHistory(PAGE_SIZE, history.items.length)
      setHistory((h) => ({ total: next.total, items: [...h.items, ...next.items] }))
      loadedCount.current = history.items.length + next.items.length
    } catch (err) {
      setError(getErrorMessage(err))
    } finally { setLoadingMore(false) }
  }

  const onCleared = () => { loadedCount.current = PAGE_SIZE; load() }
  const hasData = stats && stats.all_time_analyses > 0

  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="Analytics" title="Dashboard"
        description="Every text and image analysis run on this server is recorded here. Updates every 30 seconds."
        actions={
          <>
            <div className="segmented" role="group" aria-label="Time range">
              {RANGES.map((r) => (
                <button key={r.days} type="button" aria-pressed={days === r.days} onClick={() => setDays(r.days)}>Last {r.label}</button>
              ))}
            </div>
            <button type="button" className="btn-secondary btn-sm" onClick={load} disabled={refreshing}>
              <RefreshCw size={14} className={refreshing ? 'animate-spin' : ''} /> Refresh
            </button>
          </>
        } />
      <ApiOfflineBanner />

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
          className={`flex flex-col gap-6 transition-opacity ${refreshing ? 'opacity-70' : ''}`}>
          <div className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <StatTile label="Analyses" value={stats.total_analyses.toLocaleString()} icon={Activity}
              sub={`${stats.text_analyses} text · ${stats.image_analyses} image · last ${days} days`} />
            <StatTile label="AI-generated" value={stats.ai_content_detected.toLocaleString()} icon={AlertTriangle}
              sub={`${pct(stats.ai_content_detected, stats.total_analyses)}% of analyses`} />
            <StatTile label="Human / real" value={stats.verified_real.toLocaleString()} icon={CheckCircle2}
              sub={`${pct(stats.verified_real, stats.total_analyses)}% of analyses`} />
            <StatTile label="Average confidence" value={stats.avg_confidence == null ? '—' : `${stats.avg_confidence}%`} icon={Gauge}
              sub="In the predicted verdict" />
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
            <DailyChart daily={stats.daily} days={days} />
            <Breakdown stats={stats} />
          </div>

          <HistoryList items={history.items} total={history.total} onLoadMore={loadMore}
            loadingMore={loadingMore} onCleared={onCleared} />
        </motion.div>
      )}
    </main>
  )
}
