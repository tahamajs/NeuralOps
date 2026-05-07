import { cn } from '../../lib/utils';

interface InputProps extends React.InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  error?: string;
}

export const Input = ({ label, error, className, ...props }: InputProps) => (
  <div className="space-y-1">
    {label && <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">{label}</label>}
    <input
      className={cn(
        'w-full bg-white/5 border border-white/10 rounded-xl px-4 py-3 text-white placeholder-slate-500 focus:outline-none focus:border-neon-blue/50 transition-colors',
        error && 'border-red-500',
        className
      )}
      {...props}
    />
    {error && <p className="text-red-400 text-xs">{error}</p>}
  </div>
);