import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  Shield, Zap, Eye, Brain, FileText, Image, ArrowRight,
  CheckCircle, Globe, Lock, BarChart3, Cpu, Layers
} from 'lucide-react'

const fadeUp = {
  hidden: { opacity: 0, y: 40 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.6, delay: i * 0.1, ease: 'easeOut' } }),
}

const stats = [
  { label: 'Detection Accuracy', value: '97.4%', color: '#22d3ee' },
  { label: 'Models Deployed', value: '5', color: '#a78bfa' },
  { label: 'Datasets Used', value: '7', color: '#fb7185' },
  { label: 'Avg Latency', value: '<2s', color: '#22d3ee' },
]

const features = [
  {
    icon: FileText, title: 'AI Text Detection',
    desc: 'Detect AI-generated text using RoBERTa + LinguaForge-X ensemble with token-level explainability.',
    color: '#22d3ee', tag: 'NLP',
  },
  {
    icon: Globe, title: 'Misinformation Analysis',
    desc: 'Identify fake news and coordinated misinformation using LIAR + HC3 fine-tuned classifiers.',
    color: '#a78bfa', tag: 'BERT',
  },
  {
    icon: Image, title: 'Deepfake Image Detection',
    desc: 'EfficientNet + ViT models trained on CIFAKE & DiffusionDB to detect AI-generated images.',
    color: '#fb7185', tag: 'CV',
  },
  {
    icon: Brain, title: 'Explainable AI (XAI)',
    desc: 'Every prediction includes SHAP, LIME, GradCAM explanations with token and pixel-level heatmaps.',
    color: '#22d3ee', tag: 'XAI',
  },
  {
    icon: Layers, title: 'Ensemble Learning',
    desc: 'Weighted prediction fusion from RoBERTa and LinguaForge-X for significantly improved accuracy.',
    color: '#a78bfa', tag: 'Ensemble',
  },
  {
    icon: Cpu, title: 'Real-Time Inference',
    desc: 'FastAPI backend with optimized PyTorch inference serving results in under 2 seconds.',
    color: '#fb7185', tag: 'FastAPI',
  },
]

const pipeline = [
  { step: '01', title: 'Input', desc: 'Text or image uploaded', icon: Zap },
  { step: '02', title: 'Preprocess', desc: 'Tokenize / normalize', icon: Layers },
  { step: '03', title: 'Inference', desc: 'Ensemble model runs', icon: Cpu },
  { step: '04', title: 'XAI', desc: 'SHAP / GradCAM generated', icon: Eye },
  { step: '05', title: 'Result', desc: 'Score + explanation shown', icon: CheckCircle },
]

export default function Landing() {
  return (
    <main className="relative z-10 pt-20">
      {/* ── Hero ───────────────────────────────────────── */}
      <section className="min-h-screen flex flex-col items-center justify-center text-center px-6 py-24 relative">
        {/* Grid bg */}
        <div className="absolute inset-0 bg-grid-pattern opacity-30 pointer-events-none" />

        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0}
          className="mb-6 inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold"
          style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.2)', color: '#22d3ee' }}>
          <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
          SDG 16 Aligned · Peace, Justice & Strong Institutions
        </motion.div>

        <motion.h1 variants={fadeUp} initial="hidden" animate="show" custom={1}
          className="text-5xl md:text-7xl lg:text-8xl font-black tracking-tight leading-none mb-6 text-balance">
          <span className="text-white">Detect</span>{' '}
          <span className="gradient-text">Deepfakes.</span>
          <br />
          <span className="text-white">Fight</span>{' '}
          <span className="gradient-text-rose">Misinformation.</span>
        </motion.h1>

        <motion.p variants={fadeUp} initial="hidden" animate="show" custom={2}
          className="text-lg md:text-xl text-slate-400 max-w-2xl leading-relaxed mb-10">
          A multimodal AI platform combining <strong className="text-white">NLP</strong>,{' '}
          <strong className="text-white">Computer Vision</strong>, and{' '}
          <strong className="text-white">Explainable AI</strong> to classify deepfakes,
          AI-generated content, and fake news — with full reasoning transparency.
        </motion.p>

        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={3}
          className="flex flex-col sm:flex-row gap-4">
          <Link to="/text-analysis" className="btn-primary flex items-center gap-2 text-base px-8 py-4">
            <FileText size={16} /> Analyze Text
          </Link>
          <Link to="/image-analysis" className="btn-ghost flex items-center gap-2 text-base px-8 py-4">
            <Image size={16} /> Analyze Image
          </Link>
          <Link to="/dashboard" className="btn-ghost flex items-center gap-2 text-base px-8 py-4">
            <BarChart3 size={16} /> Dashboard
          </Link>
        </motion.div>

        {/* Floating shield */}
        <motion.div
          animate={{ y: [0, -16, 0] }}
          transition={{ duration: 5, repeat: Infinity, ease: 'easeInOut' }}
          className="mt-20 relative"
        >
          <div className="w-32 h-32 rounded-3xl flex items-center justify-center mx-auto glow-cyan"
            style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.15), rgba(124,58,237,0.15))', border: '1px solid rgba(34,211,238,0.3)' }}>
            <Shield size={56} className="text-cyan-400" />
          </div>
        </motion.div>
      </section>

      {/* ── Stats ──────────────────────────────────────── */}
      <section className="py-16 px-6">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {stats.map((s, i) => (
              <motion.div key={s.label} variants={fadeUp} initial="hidden" whileInView="show"
                viewport={{ once: true }} custom={i}
                className="card text-center group cursor-default">
                <p className="text-4xl font-black mb-1" style={{ color: s.color }}>{s.value}</p>
                <p className="text-xs text-slate-500 font-medium">{s.label}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Features ───────────────────────────────────── */}
      <section className="py-24 px-6">
        <div className="max-w-7xl mx-auto">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
            className="text-center mb-16">
            <p className="section-label mb-3">Core Capabilities</p>
            <h2 className="text-4xl md:text-5xl font-bold text-white mb-4">
              Everything you need to<br /><span className="gradient-text">fight synthetic media</span>
            </h2>
            <p className="text-slate-400 max-w-xl mx-auto">
              Six integrated detection modules powered by state-of-the-art deep learning.
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((f, i) => (
              <motion.div key={f.title} variants={fadeUp} initial="hidden" whileInView="show"
                viewport={{ once: true }} custom={i * 0.5}
                className="card group cursor-default hover:glow-cyan">
                <div className="flex items-start justify-between mb-4">
                  <div className="w-11 h-11 rounded-xl flex items-center justify-center"
                    style={{ background: `${f.color}18`, border: `1px solid ${f.color}30` }}>
                    <f.icon size={20} style={{ color: f.color }} />
                  </div>
                  <span className="tag" style={{ background: `${f.color}15`, color: f.color }}>{f.tag}</span>
                </div>
                <h3 className="text-white font-semibold text-lg mb-2">{f.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{f.desc}</p>
                <div className="mt-4 flex items-center gap-1 text-xs font-medium opacity-0 group-hover:opacity-100 transition-opacity"
                  style={{ color: f.color }}>
                  Learn more <ArrowRight size={12} />
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── Pipeline ───────────────────────────────────── */}
      <section className="py-24 px-6">
        <div className="max-w-6xl mx-auto">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
            className="text-center mb-16">
            <p className="section-label mb-3">How It Works</p>
            <h2 className="text-4xl font-bold text-white mb-4">
              From input to <span className="gradient-text">explainable insight</span>
            </h2>
          </motion.div>

          <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
            {pipeline.map((p, i) => (
              <motion.div key={p.step} variants={fadeUp} initial="hidden" whileInView="show"
                viewport={{ once: true }} custom={i * 0.15}
                className="flex flex-col items-center text-center flex-1">
                <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-3 relative"
                  style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.2)' }}>
                  <p.icon size={22} className="text-cyan-400" />
                  <span className="absolute -top-2 -right-2 text-[10px] font-bold bg-cyan-400 text-dark-950 px-1.5 py-0.5 rounded-full">
                    {p.step}
                  </span>
                </div>
                <p className="text-white font-semibold text-sm">{p.title}</p>
                <p className="text-slate-500 text-xs mt-1">{p.desc}</p>
                {i < pipeline.length - 1 && (
                  <div className="hidden md:block absolute right-0 top-1/2 text-slate-700">
                    <ArrowRight size={16} />
                  </div>
                )}
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────── */}
      <section className="py-24 px-6">
        <div className="max-w-4xl mx-auto text-center">
          <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }}
            className="card glow-cyan py-16 rounded-3xl relative overflow-hidden"
            style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.08), rgba(124,58,237,0.08))' }}>
            <div className="orb-cyan absolute top-0 left-1/2 -translate-x-1/2 -translate-y-1/2 opacity-60" />
            <Lock size={40} className="text-cyan-400 mx-auto mb-6" />
            <h2 className="text-4xl font-bold text-white mb-4">
              Ready to detect <span className="gradient-text">synthetic content?</span>
            </h2>
            <p className="text-slate-400 mb-8">
              Paste any text or upload any image and get a detailed AI analysis with full explainability in seconds.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/text-analysis" className="btn-primary flex items-center gap-2 justify-center px-8 py-4">
                <FileText size={16} /> Try Text Analysis
              </Link>
              <Link to="/image-analysis" className="btn-ghost flex items-center gap-2 justify-center px-8 py-4">
                <Image size={16} /> Try Image Analysis
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
    </main>
  )
}
