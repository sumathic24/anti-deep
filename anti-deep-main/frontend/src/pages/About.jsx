import { motion } from 'framer-motion'
import { Shield, Database, Cpu, BookOpen, Globe, Award } from 'lucide-react'

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  show: (i = 0) => ({ opacity: 1, y: 0, transition: { duration: 0.5, delay: i * 0.08 } }),
}

const datasets = [
  { name: 'LIAR Dataset', type: 'Text · Fake News', samples: '12,791', color: '#22d3ee' },
  { name: 'HC3 Dataset', type: 'Text · AI vs Human', samples: '58,000+', color: '#a78bfa' },
  { name: 'GPT-Wiki-Intro', type: 'Text · AI Generated', samples: '150,000+', color: '#22d3ee' },
  { name: 'OpenAI Human vs AI', type: 'Text · Classification', samples: '500,000+', color: '#fb7185' },
  { name: 'CIFAKE', type: 'Image · AI vs Real', samples: '120,000', color: '#a78bfa' },
  { name: 'DiffusionDB', type: 'Image · Stable Diffusion', samples: '2,000,000+', color: '#22d3ee' },
  { name: 'AI Generated Images', type: 'Image · Multi-GAN', samples: '80,000+', color: '#fb7185' },
]

const models = [
  { name: 'RoBERTa', role: 'AI Text Detection', type: 'NLP Transformer', acc: '95.8%', color: '#22d3ee' },
  { name: 'LinguaForge-X', role: 'Linguistic Pattern Analysis', type: 'Custom BERT', acc: '94.6%', color: '#a78bfa' },
  { name: 'Ensemble (R+L)', role: 'Final Text Verdict', type: 'Weighted Fusion', acc: '97.4%', color: '#fb7185' },
  { name: 'BERT Fake News', role: 'Misinformation Detection', type: 'Fine-tuned BERT', acc: '93.2%', color: '#22d3ee' },
  { name: 'EfficientNet-B4', role: 'Image Classification', type: 'CNN', acc: '96.2%', color: '#a78bfa' },
  { name: 'ViT-B/16', role: 'Vision Transformer', type: 'Transformer', acc: '95.8%', color: '#fb7185' },
]

const xaiTools = [
  { name: 'SHAP', desc: 'SHapley Additive exPlanations — computes exact feature contributions for each prediction.', color: '#fb7185' },
  { name: 'LIME', desc: 'Local Interpretable Model-agnostic Explanations — approximates model decisions locally.', color: '#a78bfa' },
  { name: 'GradCAM', desc: 'Gradient-weighted Class Activation Mapping — highlights image regions driving predictions.', color: '#22d3ee' },
]

const techStack = [
  { layer: 'Frontend', items: ['React 18', 'Tailwind CSS', 'Framer Motion', 'Recharts', 'Axios'] },
  { layer: 'Backend', items: ['FastAPI', 'Python 3.11', 'PyTorch 2.2', 'Transformers 4.40'] },
  { layer: 'Models', items: ['RoBERTa', 'LinguaForge-X', 'EfficientNet', 'ViT-B/16', 'BERT'] },
  { layer: 'XAI', items: ['SHAP 0.45', 'LIME 0.2', 'GradCAM', 'Attention Viz'] },
  { layer: 'Database', items: ['MongoDB', 'Motor (async)'] },
  { layer: 'DevOps', items: ['Docker', 'Docker Compose', 'Nginx'] },
]

export default function About() {
  return (
    <main className="relative z-10 pt-28 pb-20 px-6">
      <div className="max-w-6xl mx-auto">
        {/* Header */}
        <motion.div variants={fadeUp} initial="hidden" animate="show" custom={0} className="mb-16 text-center">
          <p className="section-label mb-3">About the Project</p>
          <h1 className="text-5xl font-black text-white mb-4">
            <span className="gradient-text">DeepFakeShield</span>
          </h1>
          <p className="text-slate-400 max-w-3xl mx-auto leading-relaxed text-lg">
            A multimodal AI detection platform built as a Final Year Major Project for B.Tech CSE (Data Science).
            Combines NLP, Computer Vision, and Explainable AI to combat the growing threat of synthetic media and coordinated misinformation.
          </p>
          <div className="mt-6 inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-medium"
            style={{ background: 'rgba(34,211,238,0.08)', border: '1px solid rgba(34,211,238,0.2)', color: '#22d3ee' }}>
            <Globe size={14} /> Aligned with SDG 16: Peace, Justice & Strong Institutions
          </div>
        </motion.div>

        {/* Objective */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0}
          className="card mb-8"
          style={{ background: 'linear-gradient(135deg, rgba(34,211,238,0.05), rgba(124,58,237,0.05))' }}>
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{ background: 'rgba(34,211,238,0.1)', border: '1px solid rgba(34,211,238,0.2)' }}>
              <Shield size={22} className="text-cyan-400" />
            </div>
            <div>
              <h2 className="text-xl font-bold text-white mb-2">Project Objective</h2>
              <p className="text-slate-400 leading-relaxed">
                The rapid rise of AI-generated deepfakes and coordinated fake news has created a digital crisis threatening public trust,
                social stability, and democratic systems. Existing tools analyze text or media independently, failing to capture the
                complete misinformation ecosystem. DeepFakeShield builds a unified multimodal detection platform analyzing text using
                transformer-based NLP, images using CNN/ViT models, detecting misinformation patterns, and providing explainable
                reasoning — all in one interface.
              </p>
            </div>
          </div>
        </motion.div>

        {/* Models */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Cpu size={20} className="text-violet-400" /> AI Models
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {models.map((m, i) => (
              <motion.div key={m.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="p-4 rounded-xl transition-all hover:scale-[1.02]"
                style={{ background: `${m.color}06`, border: `1px solid ${m.color}20` }}>
                <div className="flex items-center justify-between mb-2">
                  <p className="font-bold text-white text-sm">{m.name}</p>
                  <span className="text-xs font-black" style={{ color: m.color }}>{m.acc}</span>
                </div>
                <p className="text-xs text-slate-300 mb-1">{m.role}</p>
                <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ background: `${m.color}15`, color: m.color }}>{m.type}</span>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Datasets */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Database size={20} className="text-cyan-400" /> Datasets
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {datasets.map((d, i) => (
              <motion.div key={d.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="p-4 rounded-xl flex items-center gap-3"
                style={{ background: `${d.color}06`, border: `1px solid ${d.color}20` }}>
                <div className="w-2 h-10 rounded-full flex-shrink-0" style={{ background: d.color }} />
                <div>
                  <p className="font-semibold text-white text-sm">{d.name}</p>
                  <p className="text-xs text-slate-400">{d.type}</p>
                  <p className="text-xs font-mono mt-0.5" style={{ color: d.color }}>{d.samples} samples</p>
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* XAI */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <BookOpen size={20} className="text-rose-400" /> Explainable AI (XAI)
          </h2>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {xaiTools.map((x, i) => (
              <motion.div key={x.name} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="card">
                <p className="text-2xl font-black mb-2" style={{ color: x.color }}>{x.name}</p>
                <p className="text-sm text-slate-400 leading-relaxed">{x.desc}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Tech Stack */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0} className="mb-10">
          <h2 className="text-2xl font-bold text-white mb-6 flex items-center gap-2">
            <Cpu size={20} className="text-violet-400" /> Technology Stack
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {techStack.map((t, i) => (
              <motion.div key={t.layer} variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={i * 0.1}
                className="card p-4">
                <p className="text-xs font-bold text-cyan-400 mb-3 uppercase tracking-wider">{t.layer}</p>
                <div className="flex flex-col gap-1.5">
                  {t.items.map(item => (
                    <span key={item} className="text-xs text-slate-400">{item}</span>
                  ))}
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* SDG */}
        <motion.div variants={fadeUp} initial="hidden" whileInView="show" viewport={{ once: true }} custom={0}
          className="card text-center py-12"
          style={{ background: 'linear-gradient(135deg, rgba(34,211,238,0.05), rgba(124,58,237,0.05))' }}>
          <Award size={40} className="text-cyan-400 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-white mb-2">SDG 16 Alignment</h2>
          <p className="text-slate-400 max-w-xl mx-auto leading-relaxed">
            DeepFakeShield directly supports <strong className="text-white">Sustainable Development Goal 16</strong> —
            Peace, Justice and Strong Institutions — by combating AI-generated misinformation that undermines
            democratic processes, public trust, and social stability worldwide.
          </p>
        </motion.div>
      </div>
    </main>
  )
}
