import { motion } from 'framer-motion'
import { Link } from 'react-router-dom'
import {
  FileText, Image, Brain, Layers, Server, BarChart3, ArrowRight,
  Upload, Workflow, Cpu, Eye, CheckCircle2,
} from 'lucide-react'
import { fadeIn } from '../components/ui'

const stats = [
  { value: '97.4%', label: 'Target text-ensemble accuracy' },
  { value: '4', label: 'Detection models' },
  { value: '6', label: 'Training datasets' },
  { value: 'SHAP + GradCAM', label: 'Explainability' },
]

const features = [
  {
    icon: FileText, title: 'AI text detection', to: '/text-analysis',
    desc: 'A RoBERTa + LinguaForge-X ensemble classifies text as AI-generated or human-written, with word-level SHAP explanations.',
  },
  {
    icon: Image, title: 'Deepfake image detection', to: '/image-analysis',
    desc: 'EfficientNet-B4 and ViT-B/16 trained on CIFAKE and DiffusionDB flag AI-generated images, with GradCAM heatmaps.',
  },
  {
    icon: Brain, title: 'Explainable by default',
    desc: 'Every verdict shows its evidence: SHAP attributions for text and GradCAM heatmaps for images — on the same screen.',
  },
  {
    icon: Layers, title: 'Ensemble learning',
    desc: 'RoBERTa and LinguaForge-X are fused by tuned weights or a learned meta-classifier for more robust verdicts.',
  },
  {
    icon: Server, title: 'Inference API',
    desc: 'A FastAPI backend serves the PyTorch models, with interactive API documentation at /docs.',
  },
  {
    icon: BarChart3, title: 'Analysis dashboard', to: '/dashboard',
    desc: 'Every analysis is recorded so you can review verdict trends and recent results over time.',
  },
]

const steps = [
  { icon: Upload, title: 'Input', desc: 'Paste text or upload an image' },
  { icon: Workflow, title: 'Preprocess', desc: 'Tokenize or resize and normalize' },
  { icon: Cpu, title: 'Inference', desc: 'Models score the content' },
  { icon: Eye, title: 'Explain', desc: 'SHAP or GradCAM evidence' },
  { icon: CheckCircle2, title: 'Verdict', desc: 'Likelihood plus explanation' },
]

export default function Landing() {
  return (
    <main className="flex-1">
      {/* ── Hero ─────────────────────────────────────── */}
      <section className="relative overflow-hidden border-b border-line">
        <div className="pointer-events-none absolute inset-0"
          style={{ background: 'radial-gradient(60% 50% at 50% 0%, rgba(34,211,238,0.10), transparent 70%)' }} />
        <div className="container-page relative py-20 text-center sm:py-28">
          <motion.p variants={fadeIn} initial="hidden" animate="show" custom={0}
            className="badge mb-6 border-line-strong bg-raised text-fg-2">
            <span className="h-1.5 w-1.5 rounded-full bg-accent" /> SDG 16 aligned · Final Year Major Project
          </motion.p>
          <motion.h1 variants={fadeIn} initial="hidden" animate="show" custom={1}
            className="text-balance mx-auto max-w-4xl text-4xl font-semibold tracking-tight text-fg sm:text-5xl lg:text-6xl">
            Detect AI-generated text and <span className="text-gradient">deepfake images</span>
          </motion.h1>
          <motion.p variants={fadeIn} initial="hidden" animate="show" custom={2}
            className="text-balance mx-auto mt-6 max-w-2xl text-base leading-relaxed text-fg-2 sm:text-lg">
            DeepFakeShield combines NLP, computer vision and explainable AI. Every verdict comes with the
            evidence behind it, so you can see <em className="not-italic text-fg">why</em> content was flagged.
          </motion.p>
          <motion.div variants={fadeIn} initial="hidden" animate="show" custom={3}
            className="mt-10 flex flex-col justify-center gap-3 sm:flex-row">
            <Link to="/text-analysis" className="btn-primary px-5 py-3">
              <FileText size={16} /> Analyze text
            </Link>
            <Link to="/image-analysis" className="btn-secondary px-5 py-3">
              <Image size={16} /> Analyze an image
            </Link>
          </motion.div>
        </div>
      </section>

      {/* ── Stats ────────────────────────────────────── */}
      <section className="border-b border-line">
        <dl className="container-page grid grid-cols-2 divide-line md:grid-cols-4 md:divide-x">
          {stats.map((s) => (
            <div key={s.label} className="flex flex-col-reverse px-2 py-8 text-center md:px-6">
              <dt className="mt-1 text-xs text-fg-3 sm:text-sm">{s.label}</dt>
              <dd className="text-2xl font-semibold tracking-tight text-fg sm:text-3xl">{s.value}</dd>
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
          {features.map((f, i) => {
            const body = (
              <>
                <div className="mb-4 flex h-10 w-10 items-center justify-center rounded-lg border border-line bg-raised">
                  <f.icon size={18} className="text-accent" />
                </div>
                <h3 className="font-semibold text-fg">{f.title}</h3>
                <p className="mt-2 text-sm leading-relaxed text-fg-2">{f.desc}</p>
                {f.to && (
                  <span className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-accent">
                    Open <ArrowRight size={14} className="transition-transform group-hover:translate-x-0.5" />
                  </span>
                )}
              </>
            )
            return (
              <motion.div key={f.title} variants={fadeIn} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i}>
                {f.to ? (
                  <Link to={f.to} className="card group block h-full p-6 transition-colors hover:border-line-strong hover:bg-raised/60">{body}</Link>
                ) : (
                  <div className="card h-full p-6">{body}</div>
                )}
              </motion.div>
            )
          })}
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
            Paste text or upload an image to get a verdict with its explanation in seconds.
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
