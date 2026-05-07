import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { ExperimentFilters, ExperimentFiltersState } from '../components/ExperimentFilters'
import { ExperimentTable } from '../../dashboard/components/ExperimentTable'  // reuse
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'
import { Button } from '../../../components/ui/Button'
import { Experiment } from '../../../types/experiment'

export default function ExperimentListPage() {
  const [filters, setFilters] = useState<ExperimentFiltersState>({ search: '', status: '', type: '' })
  const [page, setPage] = useState(1)
  const pageSize = 15

  const { data: experiments = [], isLoading } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data,
    refetchInterval: 5000,
  })

  const filtered = experiments.filter((e) => {
    if (filters.status && e.status !== filters.status) return false
    if (filters.type && e.job_type !== filters.type) return false
    if (filters.search) {
      const q = filters.search.toLowerCase()
      return (e.name || '').toLowerCase().includes(q) || e.id.toLowerCase().includes(q)
    }
    return true
  })

  const totalPages = Math.ceil(filtered.length / pageSize)
  const paged = filtered.slice((page - 1) * pageSize, page * pageSize)

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        EXPERIMENT <span className="text-neon-blue">LOG</span>
      </h1>
      <ExperimentFilters value={filters} onChange={setFilters} />
      {isLoading ? (
        <Spinner />
      ) : filtered.length === 0 ? (
        <EmptyState message={filters.search || filters.status || filters.type ? 'No matching experiments' : 'No experiments yet'} />
      ) : (
        <>
          <ExperimentTable experiments={paged} />
          {totalPages > 1 && (
            <div className="flex justify-center gap-2">
              <Button variant="ghost" size="sm" disabled={page <= 1} onClick={() => setPage(page - 1)}>
                Previous
              </Button>
              <span className="py-1.5 text-xs text-slate-400">{page} / {totalPages}</span>
              <Button variant="ghost" size="sm" disabled={page >= totalPages} onClick={() => setPage(page + 1)}>
                Next
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  )
}