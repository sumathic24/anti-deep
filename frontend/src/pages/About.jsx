import { motion } from 'framer-motion'
import { ShieldCheck, Cpu, Database, BookOpen, Layers, Award } from 'lucide-react'
import { PageHeader, fadeIn } from '../components/ui'

const models = [
  { name: 'Oxidane TMR', role: 'AI text detection (default)', type: 'RoBERTa-base, trained on RAID', source: 'Pre-trained · download 503 MB', perf: 'AUROC 99.28% on RAID (reported by its authors)' },
  { name: 'SwinV2 image detector', role: 'AI image detection (default)', type: 'SwinV2 (haywoodsloan/ai-image-detector-deploy)', source: 'Pre-trained · download 781 MB', perf: '98.1% on its own validation set (reported by its author)' },
  { name: 'RoBERTa', role: 'AI text detection', type: 'Transformer (fine-tuned)', source: 'Train with training/', perf: 'Target 95.8% — not yet measured' },
  { name: 'LinguaForge-X', role: 'Linguistic pattern analysis', type: 'BERT + linguistic features', source: 'Train with training/', perf: 'Target 94.6% — not yet measured' },
  { name: 'Ensemble (R + L)', role: 'Final text verdict', type: 'Weighted fusion / meta-learner', source: 'Train with training/', perf: 'Target 97.4% — not yet measured' },
  { name: 'EfficientNet-B4', role: 'AI image detection + GradCAM', type: 'CNN', source: 'Train with training/', perf: 'Target 96.2% — not yet measured' },
  { name: 'ViT-B/16', role: 'AI image detection', type: 'Vision transformer', source: 'Train with training/', perf: 'Target 95.8% — not yet measured' },
]

const datasets = [
  { name: 'RAID', type: 'Text · AI vs human, incl. adversarial attacks', samples: '6,000,000+', usedBy: 'Pre-trained text detector' },
  { name: 'HC3', type: 'Text · AI vs human', samples: '58,000+', usedBy: 'Own text models' },
  { name: 'GPT-Wiki-Intro', type: 'Text · AI-generated', samples: '150,000+', usedBy: 'Own text models' },
  { name: 'OpenAI Human vs AI', type: 'Text · classification', samples: '500,000+', usedBy: 'Own text models' },
  { name: 'CIFAKE', type: 'Image · AI vs real', samples: '120,000', usedBy: 'Own image models' },
  { name: 'DiffusionDB', type: 'Image · Stable Diffusion', samples: '2,000,000+', usedBy: 'Own image models' },
  { name: 'AI Generated Images', type: 'Image · multi-GAN', samples: '80,000+', usedBy: 'Own image models' },
]

const xaiTools = [
  { name: 'SHAP', desc: 'SHapley Additive exPlanations — computes how many percentage points each word moved a text prediction. Works with every text model.' },
  { name: 'GradCAM', desc: 'Gradient-weighted Class Activation Mapping — highlights the image regions behind an AI-generated verdict (project’s own EfficientNet).' },
  { name: 'Model transparency', desc: 'Every result names the exact model that produced it, so a verdict from a fallback model is never mistaken for the full pipeline.' },
]

const techStack = [
  { layer: 'Frontend', items: ['React 18', 'Tailwind CSS', 'Framer Motion', 'Recharts', 'Axios'] },
  { layer: 'Backend', items: ['FastAPI', 'Python 3.11', 'PyTorch 2.2', 'Transformers 4.40'] },
  { layer: 'Models', items: ['Oxidane TMR', 'SwinV2 image detector', 'RoBERTa', 'LinguaForge-X', 'EfficientNet', 'ViT-B/16'] },
  { layer: 'Explainability', items: ['SHAP 0.45', 'GradCAM'] },
  { layer: 'Storage', items: ['SQLite (analysis history)'] },
]

function Section({ id, icon: Icon, title, description, children }) {
  return (
    <motion.section id={id} variants={fadeIn} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-12 scroll-mt-24">
      <div className="mb-5">
        <h2 className="flex items-center gap-2 text-lg font-semibold text-fg">
          <Icon size={18} className="text-fg-3" /> {title}
        </h2>
        {description && <p className="mt-1 text-sm text-fg-3">{description}</p>}
      </div>
      {children}
    </motion.section>
  )
}

export default function About() {
  return (
    <main className="container-page flex-1 py-10 sm:py-12">
      <PageHeader eyebrow="About the project" title="DeepFakeShield"
        description="A multimodal detection platform built as a Final Year Major Project for B.Tech CSE (Data Science). It combines NLP, computer vision and explainable AI to counter synthetic media." />

      <motion.div variants={fadeIn} initial="hidden" animate="show" custom={1} className="card mb-12 flex gap-4 p-6">
        <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg border border-line bg-raised">
          <ShieldCheck size={18} className="text-accent" />
        </div>
        <div>
          <h2 className="font-semibold text-fg">Project objective</h2>
          <p className="mt-2 text-sm leading-relaxed text-fg-2">
            The rapid rise of AI-generated text and deepfake images threatens public trust, social stability and
            democratic systems. Existing tools analyze text or media independently. DeepFakeShield brings both into one
            platform — transformer-based NLP for text, CNN and vision-transformer models for images — and shows the
            reasoning behind every verdict.
          </p>
        </div>
      </motion.div>

      <Section id="models" icon={Cpu} title="Models"
        description="A downloaded pre-trained detector is used first, then the project's own trained models, then a simple fallback. “Target” figures are goals for the project's own models, not measured results.">
        <div className="card overflow-x-auto">
          <table className="table-base min-w-[48rem]">
            <thead><tr><th>Model</th><th>Role</th><th>Architecture</th><th>How to get it</th><th>Performance</th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td className="font-medium text-fg">{m.name}</td>
                  <td>{m.role}</td>
                  <td>{m.type}</td>
                  <td>{m.source}</td>
                  <td className="text-fg">{m.perf}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="datasets" icon={Database} title="Datasets">
        <div className="card overflow-x-auto">
          <table className="table-base min-w-[40rem]">
            <thead><tr><th>Dataset</th><th>Modality</th><th className="text-right">Samples</th><th>Used by</th></tr></thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.name}>
                  <td className="font-medium text-fg">{d.name}</td>
                  <td>{d.type}</td>
                  <td className="tabular text-right">{d.samples}</td>
                  <td>{d.usedBy}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section id="explainability" icon={BookOpen} title="Explainable AI">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {xaiTools.map((x) => (
            <div key={x.name} className="card p-5">
              <p className="font-semibold text-fg">{x.name}</p>
              <p className="mt-2 text-sm leading-relaxed text-fg-2">{x.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section id="stack" icon={Layers} title="Technology stack">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
          {techStack.map((t) => (
            <div key={t.layer} className="card p-4">
              <p className="mb-3 text-xs font-medium uppercase tracking-wider text-fg-3">{t.layer}</p>
              <ul className="space-y-1.5 text-sm text-fg-2">
                {t.items.map((item) => <li key={item}>{item}</li>)}
              </ul>
            </div>
          ))}
        </div>
      </Section>

      <motion.section variants={fadeIn} initial="hidden" whileInView="show" viewport={{ once: true }}
        className="card flex flex-col items-center px-6 py-10 text-center">
        <Award size={28} className="mb-4 text-accent" />
        <h2 className="text-lg font-semibold text-fg">SDG 16 alignment</h2>
        <p className="mt-2 max-w-xl text-sm leading-relaxed text-fg-2">
          DeepFakeShield supports <span className="text-fg">UN Sustainable Development Goal 16</span> — Peace, Justice and
          Strong Institutions — by countering AI-generated misinformation that undermines democratic processes and public trust.
        </p>
      </motion.section>
    </main>
  )
}
