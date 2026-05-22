export function PromptEditor({ value, onChange }: any) {
  return <textarea rows={8} value={value} onChange={e => onChange(e.target.value)} className="w-full bg-black/30 border border-white/10 rounded p-3 text-white font-mono text-sm" placeholder="Enter your prompt..." />
}