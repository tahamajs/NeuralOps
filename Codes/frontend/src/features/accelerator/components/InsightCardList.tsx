import { InsightCard } from './InsightCard'
import { Spinner } from '../../../components/ui/Spinner'
import { Lightbulb } from 'lucide-react'
import type { ExperimentInsight } from '../types'

export function InsightCardList({
  insights,
  isLoading,
  onInsightDismiss,
}: {
  insights: ExperimentInsight[]
  isLoading: boolean
  onInsightDismiss?: (id: string) => void
}) {
  if (isLoading) return <Spinner size="sm" />
  if (!insights.length)
    return (
      <div className="text-center text-slate-500 py-8 flex flex-col items-center gap-2">
        <Lightbulb size={32} className="opacity-30" />
        <p>No new insights yet.</p>
        <p className="text-xs">Run more experiments to get AI‑powered suggestions.</p>
      </div>
    )
  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center">
        <h2 className="text-sm font-bold text-white flex items-center gap-2"><Lightbulb size={16} className="text-yellow-400" /> AI Insights</h2>
        <span className="text-xs text-slate-500">{insights.length} new</span>
      </div>
      {insights.map((insight) => (
        <InsightCard key={insight.id} insight={insight} onDismiss={() => onInsightDismiss?.(insight.id)} />
      ))}
    </div>
  )
}