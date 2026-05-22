// src/features/experimentExplorer/components/CorrelationMatrix.tsx
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { ParameterMetricCorrelation } from '../types'

export function CorrelationMatrix({ correlations, targetMetric }: { correlations: ParameterMetricCorrelation[]; targetMetric: string }) {
  if (!correlations?.length) return <div className="text-center text-slate-500 py-8">Run more experiments to see correlation data</div>

  const getColorClass = (value: number) => {
    if (value > 0.7) return 'text-green-400 bg-green-500/10'
    if (value > 0.3) return 'text-yellow-400 bg-yellow-500/10'
    if (value < -0.7) return 'text-red-400 bg-red-500/10'
    if (value < -0.3) return 'text-orange-400 bg-orange-500/10'
    return 'text-slate-400 bg-white/5'
  }

  return (
    <div className="space-y-2 max-h-[450px] overflow-y-auto">
      {correlations.map(c => (
        <div key={c.parameter} className="flex justify-between items-center p-2 rounded">
          <span className="text-sm text-white">{c.parameter}</span>
          <div className="flex items-center gap-2">
            <div className="w-32 h-2 bg-white/10 rounded-full overflow-hidden">
              <div className="h-full bg-neon-blue rounded-full" style={{ width: `${Math.abs(c.correlation) * 100}%`, backgroundColor: c.correlation > 0 ? '#3b82f6' : '#ef4444' }} />
            </div>
            <span className={`px-2 py-0.5 rounded text-xs font-mono ${getColorClass(c.correlation)}`}>
              {c.correlation > 0 ? '+' : ''}{c.correlation.toFixed(3)}
            </span>
          </div>
        </div>
      ))}
    </div>
  )
}