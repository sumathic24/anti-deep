import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  FileText, Image, Brain, Layers, Server, BarChart3, ArrowRight, ArrowUpRight,
  Upload, Workflow, Cpu, Eye, CheckCircle2,
} from 'lucide-react'
import { fadeIn, VerdictBadge, SplitBar } from '../components/ui'
import { useTheme } from '../utils/theme'
import { API_BASE } from '../utils/api'

const stats = [
  { value: 'Text + Images', label: 'Two detection modes' },
  { value: 'RAID', label: 'Benchmark the text detector was trained on' },
  { value: '4,803', label: 'AI generators in the image detector’s training data' },
  { value: 'SHAP + GradCAM', label: 'Explainability' },
]

const features = [
  {
    icon: FileText, title: 'AI text detection', to: '/text-analysis', cta: 'Analyze text',
    desc: 'A pre-trained RoBERTa detector — or the project’s own RoBERTa + LinguaForge-X ensemble — classifies text as AI-generated or human-written.',
  },
  {
    icon: Image, title: 'Deepfake image detection', to: '/image-analysis', cta: 'Analyze an image',
    desc: 'A Community Forensics ViT trained on thousands of generators — or the project’s own EfficientNet + ViT — flags AI-generated images.',
  },
  {
    icon: Brain, title: 'Explainable by default', to: '/about#explainability', cta: 'How it explains',
    desc: 'Verdicts show their evidence: SHAP word attributions for text and GradCAM heatmaps for images, on the same screen.',
  },
  {
    icon: Layers, title: 'Ensemble learning', to: '/about#models', cta: 'See the models',
    desc: 'The project’s RoBERTa and LinguaForge-X are fused by tuned weights or a learned meta-classifier.',
  },
  {
    icon: Server, title: 'Inference API', href: `${API_BASE}/docs`, cta: 'Open API docs',
    desc: 'A FastAPI backend serves the models, with interactive API documentation you can try in the browser.',
  },
  {
    icon: BarChart3, title: 'Analysis dashboard', to: '/dashboard', cta: 'Open dashboard',
    desc: 'Every analysis is recorded so you can review verdict trends and past results over time.',
  },
]

const steps = [
  { icon: Upload, title: 'Input', desc: 'Paste text or upload an image' },
  { icon: Workflow, title: 'Preprocess', desc: 'Tokenize or resize and normalize' },
  { icon: Cpu, title: 'Inference', desc: 'Models score the content' },
  { icon: Eye, title: 'Explain', desc: 'SHAP or GradCAM evidence' },
  { icon: CheckCircle2, title: 'Verdict', desc: 'Likelihood plus explanation' },
]

// Static illustration of a result card — clearly labelled as an example, not live output.
const EXAMPLE_TOKENS = [
  ['Furthermore,', 'ai', 0.9], ['leveraging', 'ai', 0.8], ['these', null], ['insights', 'ai', 0.4], ['enables', 'ai', 0.6],
  ['organizations', null], ['to', null], ['seamlessly', 'ai', 1], ['optimize', 'ai', 0.7], ['their', null],
  ['workflows,', null], ['honestly', 'human', 0.5], ['a', null], ['lot', 'human', 0.3], ['faster.', null],
]

function ExampleResult() {
  const { colors } = useTheme()
  const marks = { ai: colors.ai, human: colors.human }
  return (
    <div className="card relative p-5 text-left shadow-2xl sm:p-6">
      <span className="badge absolute -top-3 left-5 border-line-strong bg-raised text-fg-3">Example result</span>
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <p className="mb-2 text-xs font-medium uppercase tracking-wider text-fg-3">Verdict</p>
          <VerdictBadge isAI>AI Generated</VerdictBadge>
        </div>
        <div className="text-right">
          <p className="text-3xl font-semibold tracking-tight text-fg">92.4%</p>
          <p className="text-xs text-fg-3">confidence in this verdict</p>
        </div>
      </div>
      <SplitBar ai={0.924} />
      <p className="mb-2 mt-5 text-xs font-medium uppercase tracking-wider text-fg-3">Why — word-level SHAP</p>
      <p className="flex flex-wrap gap-1 text-sm leading-7">
        {EXAMPLE_TOKENS.map(([word, kind, strength = 0], i) => (
          <span key={i} className="rounded px-1"
            style={kind ? { background: `${marks[kind]}${Math.round((0.15 + strength * 0.5) * 255).toString(16).padStart(2, '0')}`, color: colors.fg } : { color: colors.fg2 }}>
            {word}
          </span>
        ))}
      </p>
    </div>
  )
}

function FeatureCard({ f }) {
  const body = (
    <>
      <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg border border-line bg-raised">
        <f.icon size={18} className="text-accent" />
      </div>
      <h3 className="font-semibold text-fg">{f.title}</h3>
      <p className="mt-2 flex-1 text-sm leading-relaxed text-fg-2">{f.desc}</p>
      <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-accent">
        {f.cta}
        {f.href
          ? <ArrowUpRight size={14} className="transition-transform group-hover:-translate-y-0.5 group-hover:translate-x-0.5" />
          : <ArrowRight size={14} className="transition-transform group-hover:translate-x-0.5" />}
      </span>
    </>
  )
  const cls = 'card group flex h-full flex-col p-6 transition-colors hover:border-line-strong hover:bg-raised/60'
  return f.href
    ? <a href={f.href} target="_blank" rel="noreferrer" className={cls}>{body}</a>
    : <Link to={f.to} className={cls}>{body}</Link>
}

export default function Landing() {
  return (
    <main className="flex-1">
      {/* ── Hero ─────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-line">
        <div className="pointer-events-none absolute inset-0"
          style={{ background: 'radial-gradient(60% 50% at 30% 0%, rgb(var(--accent) / 0.10), transparent 70%)' }} />
        <div className="container-page relative grid items-center gap-12 py-16 sm:py-24 lg:grid-cols-2">
          <div className="text-center lg:text-left">
            <motion.p variants={fadeIn} initial="hidden" animate="show" custom={0}
              className="badge mb-6 border-line-strong bg-raised text-fg-2">
              <span className="h-1.5 w-1.5 rounded-full bg-accent" /> SDG 16 aligned · Final Year Major Project
            </motion.p>
            <motion.h1 variants={fadeIn} initial="hidden" animate="show" custom={1}
              className="text-balance text-4xl font-semibold tracking-tight text-fg sm:text-5xl lg:text-[3.5rem] lg:leading-[1.1]">
              Detect AI&#8209;generated text and <span className="text-gradient">deepfake images</span>
            </motion.h1>
            <motion.p variants={fadeIn} initial="hidden" animate="show" custom={2}
              className="text-balance mx-auto mt-6 max-w-xl text-base leading-relaxed text-fg-2 sm:text-lg lg:mx-0">
              DeepFakeShield combines NLP, computer vision and explainable AI. Every verdict comes with the
              evidence behind it, so you can see <em className="not-italic text-fg">why</em> content was flagged.
            </motion.p>
            <motion.div variants={fadeIn} initial="hidden" animate="show" custom={3}
              className="mt-10 flex flex-col justify-center gap-3 sm:flex-row lg:justify-start">
              <Link to="/text-analysis" className="btn-primary px-5 py-3"><FileText size={16} /> Analyze text</Link>
              <Link to="/image-analysis" className="btn-secondary px-5 py-3"><Image size={16} /> Analyze an image</Link>
            </motion.div>
          </div>
          <motion.div variants={fadeIn} initial="hidden" animate="show" custom={4} className="mx-auto w-full max-w-lg">
            <ExampleResult />
          </motion.div>
        </div>
      </section>

      {/* ── Stats ────────────────────────────────────── */}
      <section className="border-b border-line">
        <dl className="container-page grid grid-cols-2 divide-line md:grid-cols-4 md:divide-x">
          {stats.map((s) => (
            <div key={s.label} className="flex flex-col-reverse px-2 py-8 text-center md:px-6">
              <dt className="mt-1 text-xs text-fg-3 sm:text-sm">{s.label}</dt>
              <dd className="text-xl font-semibold tracking-tight text-fg sm:text-2xl">{s.value}</dd>
            </div>
          ))}
        </dl>
      </section>

      {/* ── Capabilities ─────────────────────────────── */}
      <section className="container-page py-20 sm:py-24">
        <div className="mx-auto mb-12 max-w-2xl text-center">
          <p className="eyebrow mb-3">Capabilities</p>
          <h2 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">Built for transparent detection</h2>
          <p className="mt-4 text-fg-2">Text and image detection in one place, with the reasoning shown next to every result.</p>
        </div>
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {features.map((f, i) => (
            <motion.div key={f.title} variants={fadeIn} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i}>
              <FeatureCard f={f} />
            </motion.div>
          ))}
        </div>
      </section>

      {/* ── How it works ─────────────────────────────── */}
      <section className="border-y border-line bg-surface/40">
        <div className="container-page py-20 sm:py-24">
          <div className="mx-auto mb-12 max-w-2xl text-center">
            <p className="eyebrow mb-3">How it works</p>
            <h2 className="text-3xl font-semibold tracking-tight text-fg sm:text-4xl">From input to explained verdict</h2>
          </div>
          <ol className="relative grid grid-cols-1 gap-8 md:grid-cols-5 md:gap-4">
            <div className="absolute left-[10%] right-[10%] top-6 hidden h-px bg-line-strong md:block" aria-hidden="true" />
            {steps.map((s, i) => (
              <li key={s.title} className="relative flex items-start gap-4 md:flex-col md:items-center md:text-center">
                <div className="relative flex h-12 w-12 shrink-0 items-center justify-center rounded-full border border-line-strong bg-surface">
                  <s.icon size={18} className="text-accent" />
                  <span className="absolute -right-1 -top-1 flex h-5 w-5 items-center justify-center rounded-full bg-accent text-[10px] font-semibold text-accent-ink">
                    {i + 1}
                  </span>
                </div>
                <div className="md:mt-4">
                  <p className="font-medium text-fg">{s.title}</p>
                  <p className="mt-1 text-sm text-fg-3">{s.desc}</p>
                </div>
              </li>
            ))}
          </ol>
        </div>
      </section>

      {/* ── CTA ──────────────────────────────────────── */}
      <section className="container-page py-20 sm:py-24">
        <div className="card relative overflow-hidden px-6 py-12 text-center sm:px-12">
          <div className="pointer-events-none absolute inset-0"
            style={{ background: 'radial-gradient(50% 80% at 50% 100%, rgba(124,58,237,0.12), transparent 70%)' }} />
          <h2 className="relative text-2xl font-semibold tracking-tight text-fg sm:text-3xl">Check a piece of content now</h2>
          <p className="relative mx-auto mt-3 max-w-xl text-fg-2">
            Paste text or upload an image to get a verdict with its explanation.
          </p>
          <div className="relative mt-8 flex flex-col justify-center gap-3 sm:flex-row">
            <Link to="/text-analysis" className="btn-primary px-5 py-3"><FileText size={16} /> Analyze text</Link>
            <Link to="/image-analysis" className="btn-secondary px-5 py-3"><Image size={16} /> Analyze an image</Link>
          </div>
        </div>
      </section>
    </main>
  )
}
