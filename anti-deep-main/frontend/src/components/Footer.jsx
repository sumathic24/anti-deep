import { Link } from 'react-router-dom'
import { Shield, Code2, ExternalLink } from 'lucide-react'

export default function Footer() {
  return (
    <footer className="border-t border-white/5 mt-20">
      <div className="max-w-7xl mx-auto px-6 py-12">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-8 mb-10">
          <div className="col-span-1 md:col-span-2">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-8 h-8 rounded-xl flex items-center justify-center"
                style={{ background: 'linear-gradient(135deg, #06b6d4, #7c3aed)' }}>
                <Shield size={16} className="text-white" />
              </div>
              <span className="font-bold text-white">DeepFake<span className="gradient-text">Shield</span></span>
            </div>
            <p className="text-sm text-slate-500 leading-relaxed max-w-xs">
              AI-powered multimodal detection platform for deepfakes and misinformation. Aligned with SDG 16: Peace, Justice, and Strong Institutions.
            </p>
            <div className="flex items-center gap-3 mt-4">
              <a href="https://github.com" target="_blank" rel="noreferrer"
                className="p-2 rounded-lg border border-white/10 text-slate-400 hover:text-white hover:border-white/20 transition-all">
                <Code2 size={16} />
              </a>
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Platform</p>
            <div className="flex flex-col gap-2">
              {[['/', 'Home'], ['/dashboard', 'Dashboard'], ['/text-analysis', 'Text Analysis'], ['/image-analysis', 'Image Analysis']].map(([path, label]) => (
                <Link key={path} to={path} className="text-sm text-slate-400 hover:text-cyan-400 transition-colors">{label}</Link>
              ))}
            </div>
          </div>
          <div>
            <p className="text-xs font-semibold text-slate-500 uppercase tracking-widest mb-4">Models</p>
            <div className="flex flex-col gap-2">
              {['RoBERTa', 'LinguaForge-X', 'EfficientNet', 'Vision Transformer', 'SHAP / LIME', 'GradCAM'].map(m => (
                <span key={m} className="text-sm text-slate-400">{m}</span>
              ))}
            </div>
          </div>
        </div>
        <div className="pt-6 border-t border-white/5 flex flex-col md:flex-row items-center justify-between gap-4">
          <p className="text-xs text-slate-600">© 2024 DeepFakeShield. Major Project — CSE (Data Science).</p>
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-cyan-400 animate-pulse" />
            <span className="text-xs text-slate-500">All systems operational</span>
          </div>
        </div>
      </div>
    </footer>
  )
}
