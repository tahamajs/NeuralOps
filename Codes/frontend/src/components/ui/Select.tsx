import { cn } from '../../lib/utils'

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  options: { value: string; label: string }[]
}

export const Select = ({ label, options, className, ...props }: SelectProps) => (
  <div className="space-y-1.5">
    {label && (
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">{label}</label>
    )}
    <select
      className={cn(
        'w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white focus:outline-none focus:border-neon-blue/50 transition-colors appearance-none cursor-pointer',
        className
      )}
      {...props}
    >
      {options.map(opt => (
        <option key={opt.value} value={opt.value} className="bg-cyber-800">{opt.label}</option>
      ))}
    </select>
  </div>
)