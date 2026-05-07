import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function GlobalStatsPage() {
  const { data: stats } = useQuery({
    queryKey: ['globalStats'],
    queryFn: async () => {
      const [jobs, sys, gpu] = await Promise.all([
        api.get('/api/jobs/stats/').then(r => r.data),
        api.get('/api/system/status/').then(r => r.data),
        api.get('/api/system/gpu/').then(r => r.data),
      ])
      return { jobs, sys, gpu }
    },
    refetchInterval: 10000,
  })

  if (!stats) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white">Global Statistics</h1>
      
      {/* Hardware */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-bold text-white mb-4">System Resources</h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1"><span>CPU</span><span>{stats.sys.cpu?.percent}%</span></div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-neon-blue" style={{width: `${stats.sys.cpu?.percent}%`}} /></div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1"><span>Memory</span><span>{stats.sys.memory?.used_gb}/{stats.sys.memory?.total_gb} GB</span></div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-neon-purple" style={{width: `${stats.sys.memory?.percent}%`}} /></div>
            </div>
          </div>
        </Card>
        <Card>
          <h3 className="text-sm font-bold text-white mb-4">GPU Usage</h3>
          {stats.gpu?.devices?.map((dev: any, i: number) => (
            <div key={i} className="mb-3">
              <div className="flex justify-between text-xs mb-1"><span>{dev.name}</span><span>{dev.utilization_percent}%</span></div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden"><div className="h-full bg-neon-green" style={{width: `${dev.utilization_percent}%`}} /></div>
            </div>
          ))}
        </Card>
      </div>

      {/* Job Status Distribution */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-4">Job Distribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(stats.jobs.by_status).map(([status, count]: any) => (
            <div key={status} className="text-center p-4 glass rounded-xl">
              <div className="text-2xl font-bold text-white">{count}</div>
              <div className="text-xs uppercase tracking-wider text-slate-400">{status}</div>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}