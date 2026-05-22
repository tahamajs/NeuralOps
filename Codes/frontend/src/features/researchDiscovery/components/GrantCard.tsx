import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Calendar, Award, ExternalLink, TrendingUp } from 'lucide-react'

export function GrantCard({ grant }: any) {
  const daysLeft = Math.max(0, Math.ceil((new Date(grant.deadline).getTime() - Date.now()) / (1000 * 60 * 60 * 24)))
  return (
    <TerminalCard className="p-3"><div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><Award size={14} className="text-yellow-400" /><h3 className="font-bold text-white">{grant.title}</h3><Badge variant={grant.status === 'open' ? 'success' : 'warning'}>{grant.status}</Badge></div><div className="text-xs text-slate-400 mt-1">{grant.funding_body} · up to {grant.amount.toLocaleString()} USD</div><div className="flex flex-wrap gap-1 mt-2">{grant.tags.slice(0, 4).map(t => <Badge key={t} variant="info">{t}</Badge>)}</div><div className="flex items-center gap-4 mt-2 text-xs"><span className="flex items-center gap-1"><Calendar size={12} /> Deadline: {new Date(grant.deadline).toLocaleDateString()}</span><span className={`flex items-center gap-1 ${daysLeft < 30 ? 'text-red-400' : 'text-green-400'}`}>{daysLeft} days left</span></div><p className="text-sm text-slate-300 mt-2 line-clamp-2">{grant.description}</p></div><div className="text-right"><div className="flex items-center gap-1 text-neon-mint"><TrendingUp size={14} /> {Math.round(grant.match_score * 100)}% match</div><button onClick={() => window.open(grant.application_url, '_blank')} className="mt-2 text-xs text-neon-blue hover:underline">Apply →</button></div></div></TerminalCard>
  )
}