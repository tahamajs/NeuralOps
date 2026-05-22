import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Badge, Table } from '../../../components/ui'
import { Server, Cpu, HardDrive, Activity } from 'lucide-react'

export default function LoadBalancerStatusPage() {
  const { data: gpuStatus, isLoading: gpuLoading } = useQuery({
    queryKey: ['gpu-status'],
    queryFn: async () => (await api.get('/loadbalancer/gpu/status/')).data,
    refetchInterval: 5000,
  })
  const { data: schedulerStatus, isLoading: schedulerLoading } = useQuery({
    queryKey: ['scheduler-status'],
    queryFn: async () => (await api.get('/loadbalancer/scheduler/status/')).data,
  })
  const { data: queue, isLoading: queueLoading } = useQuery({
    queryKey: ['scheduler-queue'],
    queryFn: async () => (await api.get('/loadbalancer/scheduler/queue/')).data,
  })
  const { data: clusterNodes, isLoading: nodesLoading } = useQuery({
    queryKey: ['cluster-nodes'],
    queryFn: async () => (await api.get('/loadbalancer/cluster/nodes/')).data,
  })

  if (gpuLoading || schedulerLoading || queueLoading || nodesLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">Load Balancer & Cluster Status</h1>

      {/* GPU Overview */}
      <Card>
        <h2 className="text-sm font-bold text-white flex items-center gap-2 mb-3"><Cpu size={16} /> GPU Allocator Status</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div><div className="text-2xl">{gpuStatus?.total_gpus || 0}</div><div className="text-xs">Total GPUs</div></div>
          <div><div className="text-2xl text-green-400">{gpuStatus?.available_gpus || 0}</div><div className="text-xs">Available</div></div>
          <div><div className="text-2xl text-blue-400">{gpuStatus?.allocated_gpus || 0}</div><div className="text-xs">Allocated</div></div>
          <div><div className="text-2xl text-yellow-400">{gpuStatus?.pending_requests || 0}</div><div className="text-xs">Pending</div></div>
        </div>
      </Card>

      {/* Cluster Nodes */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Cluster Nodes</h2>
        <Table>
          <thead><tr><th>Node</th><th>Status</th><th>GPU</th><th>CPU</th><th>Memory</th><th>Jobs</th></tr></thead>
          <tbody>
            {clusterNodes?.map((node: any) => (
              <tr key={node.id}>
                <td className="font-mono">{node.name}</td>
                <td><Badge variant={node.status}>{node.status}</Badge></td>
                <td>{node.gpu_available}/{node.gpu_total}</td>
                <td>{node.cpu_usage}%</td>
                <td>{node.memory_usage}%</td>
                <td>{node.active_jobs}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      {/* Scheduler Queue */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Scheduler Queue</h2>
        <Table>
          <thead><tr><th>Job ID</th><th>Priority</th><th>GPU Required</th><th>Submitted</th><th>Status</th></tr></thead>
          <tbody>
            {queue?.map((job: any) => (
              <tr key={job.id}>
                <td className="font-mono">{job.id.slice(0,8)}</td>
                <td>{job.priority}</td>
                <td>{job.gpu_required}</td>
                <td>{new Date(job.submitted_at).toLocaleString()}</td>
                <td><Badge variant={job.state}>{job.state}</Badge></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>

      {/* Scheduler Metrics */}
      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Scheduler Metrics</h2>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
          <div><div className="text-2xl">{schedulerStatus?.avg_wait_time_seconds || 0}s</div><div className="text-xs">Avg Wait Time</div></div>
          <div><div className="text-2xl">{schedulerStatus?.throughput_per_hour || 0}</div><div className="text-xs">Throughput (jobs/h)</div></div>
          <div><div className="text-2xl">{schedulerStatus?.fairness_index?.toFixed(2) || 1.0}</div><div className="text-xs">Fairness Index</div></div>
          <div><div className="text-2xl">{schedulerStatus?.preemptions || 0}</div><div className="text-xs">Preemptions</div></div>
        </div>
      </Card>
    </div>
  )
}