import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Spinner } from '../../../components/ui/Spinner'
import { TrendingUp } from 'lucide-react'
import type { OpportunityScore } from '../types'

export function OpportunityScoreWidget({ score, isLoading }: { score?: OpportunityScore; isLoading?: boolean }) {
  if (isLoading) return <Spinner size="sm" />
  if (!score)
    return (
      <TerminalCard className="p-4 text-center text-slate-500">
        <TrendingUp size={32} className="mx-auto mb-2 opacity-30" />
        <p>Run experiments to see your opportunity score</p>
      </TerminalCard>
    )

  const breakdown = Object.entries(score.breakdown)
    .sort(([, a], [, b]) => b - a)
    .slice(0, 5)

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start">
        <div><div className="text-xs text-slate-400 uppercase tracking-wider">Opportunity Score</div><div className="text-4xl font-bold text-neon-mint">{score.total}</div><div className="text-xs text-slate-500 mt-1">Recommended focus: {score.recommended_focus}</div></div>
        <TrendingUp size={24} className="text-neon-mint opacity-70" />
      </div>
      <div className="mt-4 space-y-2">
        {breakdown.map(([key, value]) => (
          <div key={key}>
            <div className="flex justify-between text-xs">
              <span className="text-slate-400 capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="text-white">{value}</span>
            </div>
            <div className="w-full bg-white/10 rounded-full h-1.5"><div className="bg-neon-blue h-1.5 rounded-full" style={{ width: `${value}%` }} /></div>
          </div>
        ))}
      </div>
    </TerminalCard>
  )
}