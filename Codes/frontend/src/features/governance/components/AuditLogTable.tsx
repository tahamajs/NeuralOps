import { DataGrid } from '../../../components/ui/DataGrid'
import { Badge } from '../../../components/ui/Badge'
import { AuditLogEntry } from '../types'

export function AuditLogTable({ logs }: { logs: AuditLogEntry[] }) {
  const columns = [
    { accessorKey: 'timestamp', header: 'Timestamp', cell: ({ row }: any) => new Date(row.original.timestamp).toLocaleString() },
    { accessorKey: 'username', header: 'User' },
    { accessorKey: 'action', header: 'Action', cell: ({ row }: any) => <Badge variant="info">{row.original.action}</Badge> },
    { accessorKey: 'resource_type', header: 'Resource Type' },
    { accessorKey: 'resource_name', header: 'Resource Name' },
    { accessorKey: 'ip_address', header: 'IP Address' },
    { accessorKey: 'severity', header: 'Severity', cell: ({ row }: any) => <Badge variant={row.original.severity === 'critical' ? 'danger' : row.original.severity === 'warning' ? 'warning' : 'info'}>{row.original.severity}</Badge> },
  ]
  return <DataGrid data={logs} columns={columns} />
}