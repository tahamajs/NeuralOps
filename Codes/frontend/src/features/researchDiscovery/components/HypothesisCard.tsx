import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Brain, TrendingUp, Star, Target, Zap } from 'lucide-react'

export function HypothesisCard({ hypothesis, onSelect, selected }: any) {
  return (
    <TerminalCard className={`p-3 cursor-pointer ${selected ? 'border-neon-blue' : ''}`} onClick={() => onSelect(hypothesis.id)}><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><Brain size={14} className="text-neon-mint" /><h3 className="font-bold text-white">{hypothesis.title}</h3><Badge variant={hypothesis.status === 'generated' ? 'warning' : 'success'}>{hypothesis.status}</Badge></div><p className="text-sm text-slate-300 mt-1">{hypothesis.description.slice(0, 150)}...</p><div className="flex gap-3 mt-2 text-xs"><span className="flex items-center gap-1"><Star size={10} /> Scientific Merit: {hypothesis.scientific_merit}</span><span className="flex items-center gap-1"><TrendingUp size={10} /> Novelty: {hypothesis.novelty}</span><span className="flex items-center gap-1"><Target size={10} /> Feasibility: {hypothesis.feasibility}</span><span className="flex items-center gap-1"><Zap size={10} /> Impact: {hypothesis.impact}</span></div></div><div className="text-right"><div className="text-lg font-bold text-neon-mint">{hypothesis.total_score.toFixed(0)}</div><div className="text-xs text-slate-400">Total Score</div></div></div></TerminalCard>
  )
}