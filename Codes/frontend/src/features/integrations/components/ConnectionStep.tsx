export function ConnectionStep({ number, title, children, active, completed }: any) {
  return (
    <div className="flex gap-3"><div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-bold ${active ? 'bg-neon-blue text-white' : completed ? 'bg-green-500 text-white' : 'bg-white/10 text-slate-400'}`}>{completed ? '✓' : number}</div><div><div className="font-medium text-white">{title}</div><div className="text-slate-400 text-sm">{children}</div></div></div>
  )
}