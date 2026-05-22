import { Hypothesis } from '../types'
import { formatDistanceToNow } from 'date-fns'

export function HypothesisTimeline({ timeline }: { timeline: Hypothesis[] }) {
  return <div className="space-y-3">{timeline.map(h => (<div key={h.id} className="border-l-2 border-neon-blue pl-4 py-2"><div className="text-xs text-slate-400">{formatDistanceToNow(new Date(h.created_at), { addSuffix: true })}</div><div className="font-medium text-white">{h.title}</div><div className="text-xs text-slate-500">Score: {h.total_score.toFixed(0)}</div></div>))}</div>
}