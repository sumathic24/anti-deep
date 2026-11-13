import { ShieldCheck } from 'lucide-react'

export default function Logo({ size = 'md' }) {
  const box = size === 'sm' ? 'h-7 w-7' : 'h-8 w-8'
  return (
    <span className="flex items-center gap-2.5">
      <span className={`flex ${box} items-center justify-center rounded-lg`}
        style={{ background: 'linear-gradient(135deg, #22d3ee, #7c3aed)' }}>
        <ShieldCheck size={size === 'sm' ? 15 : 17} className="text-white" strokeWidth={2.25} />
      </span>
      <span className="text-[15px] font-semibold tracking-tight text-fg">
        DeepFake<span className="text-accent">Shield</span>
      </span>
    </span>
  )
}
