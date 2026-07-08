import { Timer } from "lucide-react";

export const ProgressBar = ({
  progress,
}: {
  progress: { current: number; total: number; eta: number };
}) => {
  const pct =
    progress.total > 0
      ? Math.min(100, (progress.current / progress.total) * 100)
      : 0;

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-400">
        <span>
          Step {progress.current || 0} / {progress.total || 0}
        </span>
        {progress.eta > 0 && (
          <span className="flex items-center gap-1">
            <Timer size={12} />
            ETA: {Math.round(progress.eta)}s
          </span>
        )}
      </div>
      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-neon-blue to-purple-500 transition-all duration-500 rounded-full"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
};
