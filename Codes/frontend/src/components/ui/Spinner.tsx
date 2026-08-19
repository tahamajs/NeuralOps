import { cn } from "../../lib/utils";

interface SpinnerProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  label?: string;
}

const sizeMap = {
  sm: "w-5 h-5",
  md: "w-8 h-8",
  lg: "w-12 h-12",
  xl: "w-16 h-16",
};

export const Spinner = ({ className, size = "md", label }: SpinnerProps) => (
  <div className={cn("flex flex-col items-center justify-center gap-3", className)}>
    <div className={cn("relative flex items-center justify-center", sizeMap[size])}>
      {/* Outer pulsing ring */}
      <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 animate-ping opacity-60" />
      {/* Outer gradient rotating ring */}
      <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyan-400 border-r-purple-500 animate-spin" />
      {/* Inner counter-rotating ring */}
      <div className="absolute inset-1 rounded-full border border-transparent border-b-emerald-400 border-l-blue-400 animate-spin [animation-direction:reverse] [animation-duration:1.5s]" />
      {/* Core glowing dot */}
      <div className="w-1.5 h-1.5 rounded-full bg-cyan-300 shadow-[0_0_8px_#38bdf8] animate-pulse" />
    </div>
    {label && (
      <span className="text-xs font-mono text-slate-400 tracking-wider animate-pulse">
        {label}
      </span>
    )}
  </div>
);

export const FullPageLoader = ({ message = "Initializing NeuralOps Quantum Workspace..." }: { message?: string }) => (
  <div className="fixed inset-0 z-50 bg-[#080d19]/90 backdrop-blur-xl flex flex-col items-center justify-center p-6 select-none">
    <div className="relative flex items-center justify-center w-24 h-24 mb-6">
      <div className="absolute inset-0 rounded-3xl bg-gradient-to-tr from-cyan-500/20 via-indigo-600/20 to-purple-600/20 blur-xl animate-pulse" />
      <div className="absolute inset-0 rounded-2xl border border-white/10 bg-black/40 backdrop-blur-md" />
      <div className="relative w-12 h-12">
        <div className="absolute inset-0 rounded-full border-2 border-transparent border-t-cyan-400 border-r-purple-500 animate-spin" />
        <div className="absolute inset-1.5 rounded-full border-2 border-transparent border-b-emerald-400 border-l-blue-400 animate-spin [animation-direction:reverse] [animation-duration:1.2s]" />
        <div className="absolute inset-3 rounded-full bg-gradient-to-tr from-cyan-400 to-indigo-500 shadow-[0_0_15px_#06b6d4] animate-pulse" />
      </div>
    </div>

    <h3 className="text-base font-bold text-white tracking-wide mb-1 font-display">
      NeuralOps AI Engine
    </h3>
    <p className="text-xs text-cyan-400 font-mono tracking-wider animate-pulse max-w-sm text-center">
      {message}
    </p>

    <div className="w-48 h-1 bg-white/10 rounded-full mt-6 overflow-hidden">
      <div className="h-full bg-gradient-to-r from-cyan-500 via-indigo-500 to-purple-500 rounded-full w-2/3 animate-[shimmer_2s_infinite]" />
    </div>
  </div>
);
