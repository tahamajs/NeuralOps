import { useState } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import { Filter, X } from 'lucide-react'

interface LeaderboardFiltersProps {
  onFilterChange: (filters: { status?: string; sort_by?: string; order?: 'asc' | 'desc' }) => void
  currentSortBy: string
  currentOrder: 'asc' | 'desc'
}

export function LeaderboardFilters({ onFilterChange, currentSortBy, currentOrder }: LeaderboardFiltersProps) {
  const [status, setStatus] = useState('')
  const [sortBy, setSortBy] = useState(currentSortBy)
  const [order, setOrder] = useState(currentOrder)

  const applyFilters = () => {
    onFilterChange({ status: status || undefined, sort_by: sortBy, order })
  }

  const resetFilters = () => {
    setStatus('')
    setSortBy('impact_score')
    setOrder('desc')
    onFilterChange({ status: undefined, sort_by: 'impact_score', order: 'desc' })
  }

  return (
    <TerminalCard className="p-3">
      <div className="flex flex-wrap gap-3 items-end">
        <Select
          label="Status"
          options={[
            { value: '', label: 'All' },
            { value: 'active', label: 'Active' },
            { value: 'completed', label: 'Completed' },
            { value: 'archived', label: 'Archived' },
          ]}
          value={status}
          onChange={(e) => setStatus(e.target.value)}
          className="w-32"
        />
        <Select
          label="Sort by"
          options={[
            { value: 'impact_score', label: 'Impact Score' },
            { value: 'h_index', label: 'h-index' },
            { value: 'total_citations', label: 'Citations' },
            { value: 'output_count', label: 'Outputs' },
          ]}
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value)}
          className="w-36"
        />
        <Select
          label="Order"
          options={[
            { value: 'desc', label: 'Highest first' },
            { value: 'asc', label: 'Lowest first' },
          ]}
          value={order}
          onChange={(e) => setOrder(e.target.value as 'asc' | 'desc')}
          className="w-32"
        />
        <Button size="sm" onClick={applyFilters}>
          <Filter size={14} className="mr-1" /> Apply
        </Button>
        <Button size="sm" variant="ghost" onClick={resetFilters}>
          <X size={14} className="mr-1" /> Reset
        </Button>
      </div>
    </TerminalCard>
  )
}