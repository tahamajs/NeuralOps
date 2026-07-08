import { cn } from "../../lib/utils";
import { ChevronDown } from "lucide-react";

interface SelectProps extends React.SelectHTMLAttributes<HTMLSelectElement> {
  label?: string;
  error?: string;
  options: { value: string; label: string; disabled?: boolean }[];
  placeholder?: string;
}

export const Select = ({
  label,
  error,
  options,
  placeholder,
  className,
  ...props
}: SelectProps) => (
  <div className="space-y-1.5">
    {label && (
      <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block">
        {label}
      </label>
    )}
    <div className="relative">
      <select
        className={cn(
          "w-full bg-white/5 border border-white/10 rounded-xl px-4 py-2.5 text-sm text-white",
          "focus:outline-none focus:border-neon-blue/50 transition-all appearance-none cursor-pointer",
          error && "border-red-500/50",
          className,
        )}
        {...props}
      >
        {placeholder && (
          <option value="" disabled className="bg-cyber-800 text-slate-500">
            {placeholder}
          </option>
        )}
        {options.map((opt) => (
          <option
            key={opt.value}
            value={opt.value}
            disabled={opt.disabled}
            className="bg-cyber-800 text-white"
          >
            {opt.label}
          </option>
        ))}
      </select>
      <ChevronDown
        size={16}
        className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
      />
    </div>
    {error && <p className="text-red-400 text-xs mt-1">{error}</p>}
  </div>
);
