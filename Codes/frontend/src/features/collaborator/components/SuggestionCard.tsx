import { useState } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Avatar } from '../../../components/ui/Avatar'
import { Progress } from '../../../components/ui/Progress'
import { UserPlus, X, BookOpen, Award, Users, Globe } from 'lucide-react'
import { SuggestedCollaborator, SuggestionReason } from '../types'

interface SuggestionCardProps {
  suggestion: SuggestedCollaborator
  onAccept: () => void
  onReject: () => void
}

const reasonConfig: Record<SuggestionReason, { label: string; color: string; icon: React.ReactNode }> = {
  expertise: { label: 'Expertise match', color: 'bg-green-500/10 text-green-400', icon: <Award size={12} /> },
  interest: { label: 'Research interests', color: 'bg-blue-500/10 text-blue-400', icon: <BookOpen size={12} /> },
  mutual: { label: 'Mutual connections', color: 'bg-purple-500/10 text-purple-400', icon: <Users size={12} /> },
  network: { label: 'Network proximity', color: 'bg-yellow-500/10 text-yellow-400', icon: <Globe size={12} /> },
}

export function SuggestionCard({ suggestion, onAccept, onReject }: SuggestionCardProps) {
  const [expanded, setExpanded] = useState(false)
  const p = suggestion.profile
  const score = suggestion.match_score
  const reason = reasonConfig[suggestion.reason]

  return (
    <TerminalCard className="p-4">
      <div className="flex gap-4"><Avatar src={p.avatar_url} size="lg" className="shrink-0" />
        <div className="flex-1"><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><h3 className="font-bold text-white text-lg">{p.name}</h3><span className="px-2 py-0.5 rounded-full text-xs bg-neon-mint/10 text-neon-mint">{p.availability === 'looking' ? 'Available' : p.availability === 'active' ? 'Active' : 'Busy'}</span></div><div className="text-sm text-slate-400">{p.title} · {p.institution}</div></div><div className="flex gap-2"><Button size="sm" variant="primary" onClick={onAccept}><UserPlus size={14} className="mr-1" /> Connect</Button><Button size="sm" variant="ghost" onClick={onReject}><X size={14} /></Button></div></div>
          <div className="flex flex-wrap gap-2 mt-2">{p.expertise.slice(0, 4).map(tag => <Badge key={tag}>{tag}</Badge>)}</div>
          <div className="flex gap-4 mt-3 text-sm"><div className="flex items-center gap-1"><Award size={14} className="text-slate-400" /><span>h‑index: {p.h_index}</span></div><div><span>📄 {p.publications_count}</span></div><div><span>📊 {p.citation_count.toLocaleString()} citations</span></div></div>
          <button onClick={() => setExpanded(!expanded)} className="text-xs text-neon-mint mt-2">Show match details</button>
          {expanded && (<div className="mt-3 p-3 bg-white/5 rounded-lg"><div className="grid grid-cols-2 gap-3"><div><div className="flex items-center gap-1 text-xs text-slate-400"><span className="px-1.5 py-0.5 rounded bg-green-500/10 text-green-400">Score</span> {score.overall}%</div><Progress value={score.overall} className="h-1 mt-1" /></div><div><div className="flex items-center gap-1 text-xs text-slate-400"><span className="px-1.5 py-0.5 rounded bg-blue-500/10 text-blue-400">Expertise</span> {score.expertise}%</div><Progress value={score.expertise} className="h-1 mt-1" /></div><div><div className="flex items-center gap-1 text-xs text-slate-400"><span className="px-1.5 py-0.5 rounded bg-yellow-500/10 text-yellow-400">Interest</span> {score.interest}%</div><Progress value={score.interest} className="h-1 mt-1" /></div><div><div className="flex items-center gap-1 text-xs text-slate-400"><span className="px-1.5 py-0.5 rounded bg-purple-500/10 text-purple-400">Network</span> {score.network}%</div><Progress value={score.network} className="h-1 mt-1" /></div></div>
            <div className="mt-3 text-xs text-slate-400 border-t border-white/10 pt-2"><div className="flex items-center gap-1">{reason.icon}<span className={reason.color}>{reason.label}</span></div>{suggestion.common_interests.length > 0 && <div className="mt-1">Common interests: {suggestion.common_interests.join(', ')}</div>}{suggestion.common_connections.length > 0 && <div>{suggestion.common_connections.length} mutual connections</div>}</div></div>)}
        </div>
      </div>
    </TerminalCard>
  )
}