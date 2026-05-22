import { Card } from '../../../components/ui'
import { FlaskConical, GitBranch, Library, FileText, StickyNote, ArrowUp } from 'lucide-react'
import type { PortfolioStats } from '../types'

interface StatsGridProps {
  stats?: PortfolioStats
}

export function StatsGrid({ stats }: StatsGridProps) {
  if (!stats) return <div className="text-center text-slate-400 py-12">No statistics available.</div>

  const statCards = [
    { label: 'Experiments', value: stats.total_experiments, icon: FlaskConical, color: 'text-neon-blue' },
    { label: 'Runs', value: stats.total_runs, icon: GitBranch, color: 'text-green-400' },
    { label: 'Models', value: stats.total_models_published, icon: Library, color: 'text-purple-400' },
    { label: 'Papers', value: stats.total_papers, icon: FileText, color: 'text-pink-400' },
    { label: 'Notes', value: stats.total_notes, icon: StickyNote, color: 'text-yellow-400' },
    { label: 'Upvotes', value: stats.total_upvotes_received, icon: ArrowUp, color: 'text-orange-400' },
  ]

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
      {statCards.map((card) => (
        <Card key={card.label} className="text-center p-5">
          <card.icon size={28} className={`mx-auto mb-2 ${card.color}`} />
          <div className="text-2xl font-bold text-white">{card.value.toLocaleString()}</div>
          <div className="text-xs text-slate-500 uppercase tracking-wider mt-1">{card.label}</div>
        </Card>
      ))}
    </div>
  )
}