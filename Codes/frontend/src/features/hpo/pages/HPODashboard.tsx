import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { hpoAPI } from '../services/api'
import { SweepCreator } from '../components/SweepCreator'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { DataGrid } from '../../../components/ui/DataGrid'
import { Plus, TrendingUp, CheckCircle, XCircle, Play } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function HPODashboard({ onClose }: { onClose?: () => void }) {
  const [showCreator, setShowCreator] = useState(false)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: sweepsData, isLoading } = useQuery({
    queryKey: ['sweeps'],
    queryFn: () => hpoAPI.getSweeps(),
  })

  const deleteSweep = useMutation({
    mutationFn: (id: string) => hpoAPI.deleteSweep(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['sweeps'] })
      success('Sweep deleted')
    },
    onError: () => error('Failed to delete sweep'),
  })

  const columns = [
    { accessorKey: 'name', header: 'Name', cell: ({ row }: any) => <Link to={`/sweeps/${row.original.id}`} className="text-neon-blue hover:underline">{row.original.name}</Link> },
    { accessorKey: 'algorithm', header: 'Algorithm', cell: ({ row }: any) => <Badge variant="info">{row.original.algorithm}</Badge> },
    { accessorKey: 'status', header: 'Status', cell: ({ row }: any) => <Badge variant={row.original.status === 'completed' ? 'success' : row.original.status === 'running' ? 'info' : 'warning'}>{row.original.status}</Badge> },
    { accessorKey: 'total_trials', header: 'Trials' },
    { accessorKey: 'best_value', header: 'Best Value', cell: ({ row }: any) => row.original.best_value?.toFixed(4) || '-' },
    { accessorKey: 'created_at', header: 'Created', cell: ({ row }: any) => new Date(row.original.created_at).toLocaleDateString() },
    { id: 'actions', header: '', cell: ({ row }: any) => (
      <div className="flex gap-2">
        <button onClick={() => deleteSweep.mutate(row.original.id)} className="text-red-400 hover:text-red-300">Delete</button>
      </div>
    ) },
  ]

  const sweeps = sweepsData?.results || []

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white">HPO Sweeps</h1>
        <div className="flex gap-2">
          <Button size="sm" onClick={() => setShowCreator(true)}><Plus size={14} className="mr-1" /> New Sweep</Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        {isLoading ? <Spinner /> : <TerminalCard noPadding><DataGrid data={sweeps} columns={columns} /></TerminalCard>}
      </div>
      <SweepCreator open={showCreator} onClose={() => setShowCreator(false)} />
    </div>
  )
}