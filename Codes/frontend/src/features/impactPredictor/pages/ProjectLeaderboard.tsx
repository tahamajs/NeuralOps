import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { impactAPI } from '../services/api'
import { LeaderboardFilters } from '../components/LeaderboardFilters'
import { LeaderboardTable } from '../components/LeaderboardTable'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Trophy, Award, TrendingUp } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ProjectLeaderboard({ onClose }: { onClose?: () => void }) {
  const [sortBy, setSortBy] = useState('impact_score')
  const [order, setOrder] = useState<'asc' | 'desc'>('desc')
  const [statusFilter, setStatusFilter] = useState('')
  const [currentPage, setCurrentPage] = useState(1)
  const { success } = useToast()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['project-comparison', { sort_by: sortBy, order, status: statusFilter, page: currentPage }],
    queryFn: () =>
      impactAPI.getProjectsComparison({
        sort_by: sortBy,
        order,
        status: statusFilter || undefined,
        limit: 20,
        offset: (currentPage - 1) * 20,
      }),
  })

  const projects = data?.results || []
  const totalPages = Math.ceil((data?.count || 0) / 20)

  const handleFilterChange = (filters: { status?: string; sort_by?: string; order?: 'asc' | 'desc' }) => {
    if (filters.status !== undefined) setStatusFilter(filters.status)
    if (filters.sort_by !== undefined) setSortBy(filters.sort_by)
    if (filters.order !== undefined) setOrder(filters.order)
    setCurrentPage(1)
    refetch()
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div>
          <h1 className="text-xl font-bold text-white flex items-center gap-2">
            <Trophy size={20} className="text-yellow-400" /> Project Leaderboard
          </h1>
          <p className="text-xs text-slate-400">Ranked by AI‑predicted impact scores</p>
        </div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            <TrendingUp size={14} className="mr-1" /> Refresh
          </Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>

      <div className="flex-1 overflow-auto p-4 space-y-4">
        <LeaderboardFilters onFilterChange={handleFilterChange} currentSortBy={sortBy} currentOrder={order} />

        {isLoading ? (
          <Spinner />
        ) : projects.length === 0 ? (
          <TerminalCard className="p-8 text-center text-slate-500">
            <Award size={32} className="mx-auto mb-2 opacity-30" />
            <p>No projects found</p>
          </TerminalCard>
        ) : (
          <LeaderboardTable projects={projects} />
        )}

        {totalPages > 1 && (
          <div className="flex justify-center gap-2 mt-4">
            <Button
              size="sm"
              variant="ghost"
              disabled={currentPage === 1}
              onClick={() => setCurrentPage((p) => p - 1)}
            >
              Previous
            </Button>
            <span className="text-sm text-slate-400">
              Page {currentPage} of {totalPages}
            </span>
            <Button
              size="sm"
              variant="ghost"
              disabled={currentPage === totalPages}
              onClick={() => setCurrentPage((p) => p + 1)}
            >
              Next
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}