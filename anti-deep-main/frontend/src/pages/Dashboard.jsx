import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { BarChart3, FileText, Image, Shield, TrendingUp, Activity, AlertTriangle, CheckCircle } from 'lucide-react'
import { AreaChart, Area, BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts'
import { Link } from 'react-router-dom'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

const areaData = [
  { day: 'Mon', text: 42, image: 18 }, { day: 'Tue', text: 67, image: 32 },
  { day: 'Wed', text: 55, image: 25 }, { day: 'Thu', text: 89, image: 41 },
  { day: 'Fri', text: 73, image: 38 }, { day: 'Sat', text: 94, image: 52 },
  { day: 'Sun', text: 81, image: 44 },
]

const barData = [
  { name: 'AI Text', value: 68, color: '#22d3ee' },
  { name: 'Human', value: 32, color: '#a78bfa' },
  { name: 'Fake News', value: 45, color: '#fb7185' },
  { name: 'Real News', value: 55, color: '#22d3ee' },
  { name: 'AI Image', value: 71, color: '#fb7185' },
  { name: 'Real Image', value: 29, color: '#a78bfa' },
]

const pieData = [
  { name: 'AI-Generated Text', value: 38, color: '#22d3ee' },
  { name: 'Fake News', value: 27, color: '#fb7185' },
  { name: 'AI Image', value: 21, color: '#a78bfa' },
  { name: 'Human Content', value: 14, color: '#64748b' },
]

const recentAnalyses = [
  { type: 'text', label: '"Scientists discover new black hole…"', result: 'AI Generated', confidence: 94.2, color: '#fb7185', icon: FileText },
  { type: 'image', label: 'portrait_photo.jpg', result: 'AI Image', confidence: 88.7, color: '#fb7185', icon: Image },
  { type: 'text', label: '"Breaking: Local government passes bill…"', result: 'Fake News', confidence: 91.3, color: '#f97316', icon: FileText },
  { type: 'text', label: '"The study involved 200 participants…"', result: 'Human Written', confidence: 82.1, color: '#22d3ee', icon: FileText },
  { type: 'image', label: 'landscape_mountain.png', result: 'Real Image', confidence: 96.4, color: '#22d3ee', icon: Image },
]

const topStats = [
  { label: 'Total Analyses', value: '1,247', change: '+12%', icon: Activity, color: '#22d3ee' },
  { label: 'AI Content Detected', value: '847', change: '+8%', icon: AlertTriangle, color: '#fb7185' },
  { label: 'Avg Confidence', value: '91.3%', change: '+2.1%', icon: TrendingUp, color: '#a78bfa' },
  { label: 'Verified Real', value: '400', change: '+4%', icon: CheckCircle, color: '#22d3ee' },
]

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-strong rounded-xl px-4 py-3 text-xs">
        <p className="text-slate-400 mb-1">{label}</p>
        {payload.map((p) => (
          <p key={p.dataKey} style={{ color: p.color }}>{p.dataKey}: <strong>{p.value}</strong></p>
        ))}
      </div>
    )
  }
  return null
}

export default function Dashboard() {
  const [animatedStats, setAnimatedStats] = useState(topStats.map(() => 0))

  useEffect(() => {
    const targets = [1247, 847, 91.3, 400]
    targets.forEach((target, i) => {
      let start = 0
      const step = target / 40
      const timer = setInterval(() => {
        start = Math.min(start + step, target)
        setAnimatedStats(prev => {
          const next = [...prev]
          next[i] = start
          return next
        })
        if (start >= target) clearInterval(timer)
      }, 20)
    })
  }, [])

  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-10">
          <p className="section-label mb-2">Analytics Overview</p>
          <h1 className="text-4xl font-bold text-white mb-2">
            Detection <span className="gradient-text">Dashboard</span>
          </h1>
          <p className="text-slate-400">Real-time monitoring of all AI content detection activity.</p>
        </motion.div>

        {/* Top stats */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {topStats.map((s, i) => (
            <motion.div key={s.label} variants={fadeUp} initial="hidden" animate="show" custom={i * 0.1}
              className="card group">
              <div className="flex items-start justify-between mb-3">
                <div className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{ background: `${s.color}18`, border: `1px solid ${s.color}30` }}>
                  <s.icon size={18} style={{ color: s.color }} />
                </div>
                <span className="text-xs font-medium px-2 py-0.5 rounded-full"
                  style={{ background: 'rgba(34,211,238,0.08)', color: '#22d3ee' }}>{s.change}</span>
              </div>
              <p className="text-2xl font-bold text-white">
                {i === 2 ? animatedStats[i].toFixed(1) + '%' : Math.floor(animatedStats[i]).toLocaleString()}
              </p>
              <p className="text-xs text-slate-500 mt-1">{s.label}</p>
            </motion.div>
          ))}
        </div>

        {/* Charts row */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
          {/* Area chart */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.5}
            className="card lg:col-span-2">
            <div className="flex items-center justify-between mb-6">
              <div>
                <p className="text-white font-semibold">Analyses This Week</p>
                <p className="text-xs text-slate-500 mt-1">Text vs Image detections</p>
              </div>
              <div className="flex items-center gap-4 text-xs text-slate-400">
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-cyan-400 inline-block rounded" /> Text</span>
                <span className="flex items-center gap-1.5"><span className="w-3 h-0.5 bg-violet-400 inline-block rounded" /> Image</span>
              </div>
            </div>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={areaData}>
                <defs>
                  <linearGradient id="colorText" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22d3ee" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22d3ee" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="colorImage" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#a78bfa" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#a78bfa" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="day" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="text" stroke="#22d3ee" strokeWidth={2} fill="url(#colorText)" />
                <Area type="monotone" dataKey="image" stroke="#a78bfa" strokeWidth={2} fill="url(#colorImage)" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Pie chart */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.6} className="card">
            <p className="text-white font-semibold mb-1">Content Breakdown</p>
            <p className="text-xs text-slate-500 mb-4">All-time distribution</p>
            <ResponsiveContainer width="100%" height={160}>
              <PieChart>
                <Pie data={pieData} cx="50%" cy="50%" innerRadius={45} outerRadius={70}
                  paddingAngle={4} dataKey="value">
                  {pieData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip content={<CustomTooltip />} />
              </PieChart>
            </ResponsiveContainer>
            <div className="space-y-2 mt-2">
              {pieData.map(d => (
                <div key={d.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                    <span className="text-slate-400">{d.name}</span>
                  </div>
                  <span className="text-white font-medium">{d.value}%</span>
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Bar chart + recent */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.7} className="card">
            <p className="text-white font-semibold mb-1">Model Detection Rate</p>
            <p className="text-xs text-slate-500 mb-6">Percentage of each content type detected</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={barData} barSize={28}>
                <XAxis dataKey="name" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="value" radius={[6, 6, 0, 0]}>
                  {barData.map((entry, index) => (
                    <Cell key={index} fill={entry.color} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Recent analyses */}
          <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0.8} className="card">
            <div className="flex items-center justify-between mb-6">
              <p className="text-white font-semibold">Recent Analyses</p>
              <Link to="/text-analysis" className="text-xs text-cyan-400 hover:text-cyan-300 transition-colors">
                New analysis →
              </Link>
            </div>
            <div className="space-y-3">
              {recentAnalyses.map((a, i) => (
                <motion.div key={i} initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: 0.9 + i * 0.08 }}
                  className="flex items-center gap-3 p-3 rounded-xl transition-colors hover:bg-white/3">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{ background: `${a.color}15` }}>
                    <a.icon size={14} style={{ color: a.color }} />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p className="text-xs text-white font-medium truncate">{a.label}</p>
                    <p className="text-[11px] text-slate-500">{a.type === 'text' ? 'Text' : 'Image'} · {a.confidence}% confidence</p>
                  </div>
                  <span className="text-[11px] font-semibold px-2 py-1 rounded-lg flex-shrink-0"
                    style={{ background: `${a.color}15`, color: a.color }}>{a.result}</span>
                </motion.div>
              ))}
            </div>
          </motion.div>
        </div>
      </div>
    </main>
  )
}
