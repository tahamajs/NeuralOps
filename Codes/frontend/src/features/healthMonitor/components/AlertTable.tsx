import { DataGrid } from '../../../components/ui/DataGrid'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Alert, AlertSeverity } from '../types'
import { CheckCircle, Bell, AlertCircle } from 'lucide-react'

export function AlertTable({ alerts, onAcknowledge, onResolve }: { alerts: Alert[]; onAcknowledge: (id: string) => void; onResolve: (id: string) => void }) {
  const severityConfig: Record<AlertSeverity, { variant: any; icon: any }> = { info: { variant: 'info', icon: AlertCircle }, warning: { variant: 'warning', icon: AlertCircle }, critical: { variant: 'danger', icon: AlertCircle }, fatal: { variant: 'danger', icon: AlertCircle } }
  const columns = [
    { accessorKey: 'created_at', header: 'Time', cell: ({ row }: any) => new Date(row.original.created_at).toLocaleString() },
    { accessorKey: 'title', header: 'Title' },
    { accessorKey: 'severity', header: 'Severity', cell: ({ row }: any) => <Badge variant={severityConfig[row.original.severity].variant}>{row.original.severity}</Badge> },
    { accessorKey: 'status', header: 'Status', cell: ({ row }: any) => <Badge variant={row.original.status === 'acknowledged' ? 'warning' : row.original.status === 'resolved' ? 'success' : 'danger'}>{row.original.status}</Badge> },
    { id: 'actions', header: '', cell: ({ row }: any) => (<div className="flex gap-1">{row.original.status === 'active' && <Button size="sm" variant="ghost" onClick={() => onAcknowledge(row.original.id)}><Bell size={12} /> Acknowledge</Button>}{row.original.status !== 'resolved' && <Button size="sm" variant="ghost" onClick={() => onResolve(row.original.id)}><CheckCircle size={12} /> Resolve</Button>}</div>) },
  ]
  return <DataGrid data={alerts} columns={columns} />
}