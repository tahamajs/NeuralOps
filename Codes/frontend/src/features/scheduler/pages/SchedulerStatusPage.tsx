import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Table, Badge } from '../../../components/ui'

export default function SchedulerStatusPage() {
  const { data: status, isLoading } = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: async () => (await api.get('/scheduler/status/')).data,
  })
  const { data: fairshare, isLoading: fsLoading } = useQuery({
    queryKey: ['fairshare'],
    queryFn: async () => (await api.get('/scheduler/fairshare/')).data,
  })
  const { data: metrics } = useQuery({
    queryKey: ['scheduler-metrics'],
    queryFn: async () => (await api.get('/scheduler/metrics/')).data,
  })
  const { data: config } = useQuery({
    queryKey: ['scheduler-config'],
    queryFn: async () => (await api.get('/scheduler/config/')).data,
  })

  if (isLoading || fsLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Scheduler Status</h1>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Current Scheduler</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div><div className="text-2xl">{status?.running_jobs || 0}</div><div>Running</div></div>
          <div><div className="text-2xl text-yellow-400">{status?.queued_jobs || 0}</div><div>Queued</div></div>
          <div><div className="text-2xl">{status?.total_cpu_cores || 0}</div><div>Total CPU Cores</div></div>
          <div><div className="text-2xl">{status?.total_gpu_devices || 0}</div><div>Total GPUs</div></div>
        </div>
      </Card>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Fair‑Share Allocation</h2>
        <Table>
          <thead><tr><th>Group</th><th>Share</th><th>Usage</th><th>Fairness</th></tr></thead>
          <tbody>
            {fairshare?.map((item: any) => (
              <tr key={item.group}>
                <td>{item.group}</td>
                <td>{item.share_percent}%</td>
                <td>{item.usage_percent}%</td>
                <td><Badge variant={item.fairness > 0.9 ? 'success' : 'warning'}>{item.fairness.toFixed(2)}</Badge></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Scheduler Configuration</h2>
        <div className="space-y-1 text-sm">
          <div>Policy: <code className="text-neon-blue">{config?.policy}</code></div>
          <div>Preemption: {config?.preemption_enabled ? 'Enabled' : 'Disabled'}</div>
          <div>Backfilling: {config?.backfilling ? 'Yes' : 'No'}</div>
        </div>
      </Card>
    </div>
  )
}