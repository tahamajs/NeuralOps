export default function ProgressBar({ progress }: { progress: { current: number; total: number; eta: number } }) {
  const pct = progress.total > 0 ? Math.min(100, (progress.current / progress.total) * 100) : 0
  return (
    <div>
      <div className="flex justify-between text-sm text-gray-600 mb-1">
        <span>{progress.current} / {progress.total} steps</span>
        {progress.eta > 0 && <span>ETA: {Math.round(progress.eta)}s</span>}
      </div>
      <div className="w-full bg-gray-200 rounded-full h-3">
        <div className="bg-blue-600 h-3 rounded-full" style={{ width: `${pct}%` }} />
      </div>
    </div>
  )
}