import { TerminalCard } from '../../../components/ui/TerminalCard'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'

export function MetricWidget({ data, metricKey, config }: any) {
  const latest = data[data.length - 1]?.value
  const previous = data[data.length - 2]?.value
  const change = previous ? ((latest - previous) / previous) * 100 : 0
  const trendIcon = change > 0 ? <TrendingUp size={14} className="text-green-400" /> : change < 0 ? <TrendingDown size={14} className="text-red-400" /> : <Minus size={14} className="text-yellow-400" />
  return (
    <div className="text-center">
      <div className="text-2xl font-bold text-white">{latest?.toFixed(4) ?? '—'}</div>
      <div className="text-xs text-slate-400">{metricKey}</div>
      <div className="flex items-center justify-center gap-1 mt-1 text-xs">
        {trendIcon}
        <span className={change > 0 ? 'text-green-400' : change < 0 ? 'text-red-400' : 'text-yellow-400'}>
          {change > 0 ? '+' : ''}{change.toFixed(2)}%
        </span>
      </div>
    </div>
  )
}