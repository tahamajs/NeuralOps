import { cn } from "../../lib/utils";

interface SwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  disabled?: boolean;
  label?: string;
}

export const Switch = ({ checked, onChange, disabled, label }: SwitchProps) => (
  <label
    className={cn(
      "relative inline-flex items-center gap-2",
      disabled ? "cursor-not-allowed opacity-50" : "cursor-pointer",
    )}
  >
    <input
      type="checkbox"
      className="sr-only peer"
      checked={checked}
      disabled={disabled}
      onChange={(e) => onChange(e.target.checked)}
    />
    <div className="w-9 h-5 bg-cyber-600 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-neon-blue/30 rounded-full peer peer-checked:after:translate-x-full after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all after:duration-200 peer-checked:bg-neon-blue" />
    {label && <span className="text-sm text-slate-300">{label}</span>}
  </label>
);
