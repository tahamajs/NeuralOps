import { Link } from 'react-router-dom'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Avatar } from '../../../components/ui/Avatar'
import { TrendingUp, TrendingDown, Minus } from 'lucide-react'
import { ProjectComparison } from '../types'

export function LeaderboardTable({ projects }: { projects: ProjectComparison[] }) {
  const getTrendIcon = (trend: 'up' | 'down' | 'stable') => {
    switch (trend) {
      case 'up': return <TrendingUp size={14} className="text-green-400" />
      case 'down': return <TrendingDown size={14} className="text-red-400" />
      default: return <Minus size={14} className="text-yellow-400" />
    }
  }

  const getScoreColor = (score: number) => {
    if (score >= 80) return 'text-green-400'
    if (score >= 60) return 'text-yellow-400'
    return 'text-red-400'
  }

  return (
    <TerminalCard noPadding>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-slate-400 border-b border-white/5">
              <th className="px-4 py-3 w-12">#</th>
              <th className="px-4 py-3">Project</th>
              <th className="px-4 py-3 text-center">Impact Score</th>
              <th className="px-4 py-3 text-center">h-index</th>
              <th className="px-4 py-3 text-center">Citations</th>
              <th className="px-4 py-3 text-center">Outputs</th>
              <th className="px-4 py-3 text-center">Status</th>
              <th className="px-4 py-3 text-center">Trend</th>
              <th className="px-4 py-3 text-right">Actions</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-white/5">
            {projects.map((project, idx) => (
              <tr key={project.id} className="hover:bg-white/5 transition-colors">
                <td className="px-4 py-3 font-mono font-bold">{idx + 1}</td>
                <td className="px-4 py-3">
                  <div className="flex items-center gap-3">
                    <Avatar src={undefined} size="sm" />
                    <div>
                      <div className="font-medium text-white">{project.name}</div>
                      <div className="text-xs text-slate-400">{project.owner}</div>
                    </div>
                  </div>
                </td>
                <td className="px-4 py-3 text-center font-bold">
                  <span className={getScoreColor(project.impact_score)}>
                    {project.impact_score.toFixed(1)}
                  </span>
                </td>
                <td className="px-4 py-3 text-center">{project.metrics.h_index}</td>
                <td className="px-4 py-3 text-center">{project.metrics.total_citations.toLocaleString()}</td>
                <td className="px-4 py-3 text-center">{project.metrics.output_count}</td>
                <td className="px-4 py-3 text-center">
                  <Badge
                    variant={
                      project.status === 'active'
                        ? 'info'
                        : project.status === 'completed'
                        ? 'success'
                        : 'secondary'
                    }
                  >
                    {project.status}
                  </Badge>
                </td>
                <td className="px-4 py-3 text-center">
                  <div className="flex items-center justify-center gap-1">
                    {getTrendIcon(project.trend)}
                    <span className={`text-xs ${project.change > 0 ? 'text-green-400' : project.change < 0 ? 'text-red-400' : 'text-slate-400'}`}>
                      {project.change > 0 ? '+' : ''}{project.change}
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3 text-right">
                  <Link to={`/impact/${project.id}`}>
                    <Button size="sm" variant="ghost">
                      Details
                    </Button>
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </TerminalCard>
  )
}