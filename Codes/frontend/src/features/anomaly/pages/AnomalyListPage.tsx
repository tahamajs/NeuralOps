import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { anomalyAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { DataGrid } from '../../../components/ui/DataGrid'
import { AlertTriangle, CheckCircle, Eye, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function AnomalyListPage({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState('')
  const [severityFilter, setSeverityFilter] = useState('')
  const [runIdSearch, setRunIdSearch] = useState('')
  const [page, setPage] = useState(1)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['anomalies', { status: statusFilter, severity: severityFilter, run_id: runIdSearch, page }],
    queryFn: () => anomalyAPI.getAnomalies({ status: statusFilter || undefined, severity: severityFilter || undefined, run_id: runIdSearch || undefined }),
  })

  const detectMutation = useMutation({
    mutationFn: (runId: string) => anomalyAPI.detectAnomaly(runId),
    onSuccess: (_, runId) => {
      success('Anomaly detection triggered', `Run ${runId.slice(0, 8)} will be analyzed.`)
      refetch()
    },
    onError: () => error('Failed to trigger detection'),
  })

  const resolveMutation = useMutation({
    mutationFn: (anomalyId: string) => anomalyAPI.resolveAnomaly(anomalyId, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomalies'] })
      success('Anomaly marked as resolved')
    },
  })

  const columns = [
    { accessorKey: 'run_name', header: 'Run', cell: ({ row }: any) => <Link to={`/runs/${row.original.run_id}`} className="text-neon-blue hover:underline">{row.original.run_name || row.original.run_id.slice(0, 8)}</Link> },
    { accessorKey: 'anomaly_type', header: 'Type', cell: ({ row }: any) => <Badge variant="warning">{row.original.anomaly_type.replace('_', ' ')}</Badge> },
    { accessorKey: 'severity', header: 'Severity', cell: ({ row }: any) => {
      const color = { low: 'info', medium: 'warning', high: 'danger', critical: 'danger' }[row.original.severity]
      return <Badge variant={color as any}>{row.original.severity}</Badge>
    }},
    { accessorKey: 'summary', header: 'Summary' },
    { accessorKey: 'status', header: 'Status', cell: ({ row }: any) => <Badge variant={row.original.status === 'resolved' ? 'success' : 'warning'}>{row.original.status}</Badge> },
    { accessorKey: 'detected_at', header: 'Detected', cell: ({ row }: any) => new Date(row.original.detected_at).toLocaleString() },
    { id: 'actions', header: '', cell: ({ row }: any) => (
      <div className="flex gap-2">
        <Link to={`/anomalies/${row.original.id}`}><Eye size={16} className="text-slate-400 hover:text-white" /></Link>
        {row.original.status !== 'resolved' && <button onClick={() => resolveMutation.mutate(row.original.id)}><CheckCircle size={16} className="text-green-400" /></button>}
      </div>
    ) },
  ]

  if (isLoading) return <Spinner />

  const anomalies = data?.results || []
  const totalPages = Math.ceil((data?.count || 0) / 20)

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <h1 className="text-xl font-bold text-white flex items-center gap-2"><AlertTriangle size={20} className="text-yellow-400" /> Anomaly Detection</h1>
        {onClose && <button onClick={onClose} className="text-slate-400">✕</button>}
      </div>
      <div className="p-4 space-y-4 flex-1 overflow-auto">
        <div className="flex gap-3 items-end">
          <Select label="Status" options={[{ value: '', label: 'All' }, { value: 'open', label: 'Open' }, { value: 'investigating', label: 'Investigating' }, { value: 'resolved', label: 'Resolved' }]} value={statusFilter} onChange={e => setStatusFilter(e.target.value)} className="w-36" />
          <Select label="Severity" options={[{ value: '', label: 'All' }, { value: 'low', label: 'Low' }, { value: 'medium', label: 'Medium' }, { value: 'high', label: 'High' }, { value: 'critical', label: 'Critical' }]} value={severityFilter} onChange={e => setSeverityFilter(e.target.value)} className="w-36" />
          <Input placeholder="Run ID" value={runIdSearch} onChange={e => setRunIdSearch(e.target.value)} className="w-48" />
          <Button size="sm" onClick={() => refetch()}>Search</Button>
          <Button size="sm" variant="ghost" onClick={() => detectMutation.mutate(prompt('Enter Run ID to scan:') || '')}>Trigger Scan</Button>
        </div>
        <TerminalCard noPadding>
          <DataGrid data={anomalies} columns={columns} />
        </TerminalCard>
        {totalPages > 1 && <div className="flex justify-center gap-2"><button disabled={page === 1} onClick={() => setPage(p => p-1)}>Prev</button><span>{page} / {totalPages}</span><button disabled={page === totalPages} onClick={() => setPage(p => p+1)}>Next</button></div>}
      </div>
    </div>
  )
}