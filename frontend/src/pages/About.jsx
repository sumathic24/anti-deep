import { motion } from 'framer-motion'
import { ShieldCheck, Cpu, Database, BookOpen, Layers, Award } from 'lucide-react'
import { PageHeader, fadeIn } from '../components/ui'

const models = [
  { name: 'RoBERTa', role: 'AI text detection', type: 'Transformer (fine-tuned)', acc: '95.8%' },
  { name: 'LinguaForge-X', role: 'Linguistic pattern analysis', type: 'BERT + linguistic features', acc: '94.6%' },
  { name: 'Ensemble (R + L)', role: 'Final text verdict', type: 'Weighted fusion / meta-learner', acc: '97.4%' },
  { name: 'EfficientNet-B4', role: 'AI image detection', type: 'CNN', acc: '96.2%' },
  { name: 'ViT-B/16', role: 'AI image detection', type: 'Vision transformer', acc: '95.8%' },
]

const datasets = [
  { name: 'HC3', type: 'Text · AI vs human', samples: '58,000+' },
  { name: 'GPT-Wiki-Intro', type: 'Text · AI-generated', samples: '150,000+' },
  { name: 'OpenAI Human vs AI', type: 'Text · classification', samples: '500,000+' },
  { name: 'CIFAKE', type: 'Image · AI vs real', samples: '120,000' },
  { name: 'DiffusionDB', type: 'Image · Stable Diffusion', samples: '2,000,000+' },
  { name: 'AI Generated Images', type: 'Image · multi-GAN', samples: '80,000+' },
]

const xaiTools = [
  { name: 'SHAP', desc: 'SHapley Additive exPlanations — computes each word’s contribution to a text prediction.' },
  { name: 'LIME', desc: 'Local Interpretable Model-agnostic Explanations — approximates model decisions locally.' },
  { name: 'GradCAM', desc: 'Gradient-weighted Class Activation Mapping — highlights the image regions driving a prediction.' },
]

const techStack = [
  { layer: 'Frontend', items: ['React 18', 'Tailwind CSS', 'Framer Motion', 'Recharts', 'Axios'] },
  { layer: 'Backend', items: ['FastAPI', 'Python 3.11', 'PyTorch 2.2', 'Transformers 4.40'] },
  { layer: 'Models', items: ['RoBERTa', 'LinguaForge-X', 'EfficientNet', 'ViT-B/16'] },
  { layer: 'Explainability', items: ['SHAP 0.45', 'LIME 0.2', 'GradCAM'] },
  { layer: 'Storage', items: ['SQLite (analysis history)'] },
  { layer: 'DevOps', items: ['Docker', 'Docker Compose', 'Nginx'] },
]

function Section({ icon: Icon, title, description, children }) {
  return (
    <motion.section variants={fadeIn} initial="hidden" whileInView="show" viewport={{ once: true }} className="mb-12">
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

      <Section icon={Cpu} title="Models"
        description="Accuracy figures are the project's target performance; measure them on your own test split after training.">
        <div className="card overflow-x-auto">
          <table className="table-base min-w-[36rem]">
            <thead><tr><th>Model</th><th>Role</th><th>Architecture</th><th className="text-right">Target accuracy</th></tr></thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name}>
                  <td className="font-medium text-fg">{m.name}</td>
                  <td>{m.role}</td>
                  <td>{m.type}</td>
                  <td className="tabular text-right text-fg">{m.acc}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section icon={Database} title="Datasets">
        <div className="card overflow-x-auto">
          <table className="table-base min-w-[28rem]">
            <thead><tr><th>Dataset</th><th>Modality</th><th className="text-right">Samples</th></tr></thead>
            <tbody>
              {datasets.map((d) => (
                <tr key={d.name}>
                  <td className="font-medium text-fg">{d.name}</td>
                  <td>{d.type}</td>
                  <td className="tabular text-right">{d.samples}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      <Section icon={BookOpen} title="Explainable AI">
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          {xaiTools.map((x) => (
            <div key={x.name} className="card p-5">
              <p className="font-semibold text-fg">{x.name}</p>
              <p className="mt-2 text-sm leading-relaxed text-fg-2">{x.desc}</p>
            </div>
          ))}
        </div>
      </Section>

      <Section icon={Layers} title="Technology stack">
        <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-6">
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
