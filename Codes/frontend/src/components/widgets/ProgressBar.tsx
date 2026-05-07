export const ProgressBar = ({ progress }: { progress: { current: number; total: number; eta: number } }) => {
  const pct = progress.total > 0 ? Math.min(100, (progress.current / progress.total) * 100) : 0
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-slate-400">
        <span>{progress.current} / {progress.total} steps</span>
        {progress.eta > 0 && <span>ETA: {Math.round(progress.eta)}s</span>}
      </div>
      <div className="w-full h-2 bg-white/5 rounded-full overflow-hidden">
        <div
          className="h-full bg-gradient-to-r from-neon-blue to-purple-500 transition-all duration-500"
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}