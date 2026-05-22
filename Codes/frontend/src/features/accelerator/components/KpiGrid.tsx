import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Spinner } from '../../../components/ui/Spinner'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import type { KpiMetric } from '../types'

export function KpiGrid({ kpis, isLoading }: { kpis: KpiMetric[]; isLoading: boolean }) {
  if (isLoading) return <Spinner size="sm" />

  const getTrendIcon = (trend: KpiMetric['trend']) => {
    switch (trend) {
      case 'up':
        return <TrendingUp size={14} className="text-green-400" />
      case 'down':
        return <TrendingDown size={14} className="text-red-400" />
      default:
        return <Minus size={14} className="text-yellow-400" />
    }
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {kpis.map((kpi) => (
        <TerminalCard key={kpi.label} className="p-4 text-center">
          <div className="text-xs text-slate-400 uppercase tracking-wider">{kpi.label}</div>
          <div className="text-3xl font-bold text-white mt-1">
            {kpi.value.toLocaleString()} {kpi.unit}
          </div>
          <div className="flex items-center justify-center gap-1 mt-1">
            {getTrendIcon(kpi.trend)}
            <span className={kpi.change >= 0 ? 'text-green-400' : 'text-red-400'}>
              {kpi.change >= 0 ? '+' : ''}{kpi.change}%
            </span>
          </div>
          {kpi.description && <div className="text-xs text-slate-500 mt-1">{kpi.description}</div>}
        </TerminalCard>
      ))}
    </div>
  )
}