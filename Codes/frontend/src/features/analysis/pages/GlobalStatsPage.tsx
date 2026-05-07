import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import { Cpu, MemoryStick, HardDrive, Monitor } from 'lucide-react'

export default function GlobalStatsPage() {
  const { data, isLoading } = useQuery({
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

  if (isLoading) return <Spinner />
  if (!data) return <p className="text-slate-400 text-sm text-center py-8">Failed to load statistics.</p>

  const { jobs, sys, gpu } = data

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <Monitor size={24} className="text-neon-blue" />
        Global <span className="text-neon-blue">Statistics</span>
      </h1>

      {/* System Resources */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
            <Cpu size={16} className="text-neon-blue" /> System Resources
          </h3>
          <div className="space-y-4">
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">CPU ({sys.cpu?.count} cores)</span>
                <span className="text-white font-mono">{sys.cpu?.percent}%</span>
              </div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-neon-blue transition-all" style={{ width: `${sys.cpu?.percent}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Memory</span>
                <span className="text-white font-mono">{sys.memory?.used_gb} / {sys.memory?.total_gb} GB</span>
              </div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-neon-purple transition-all" style={{ width: `${sys.memory?.percent}%` }} />
              </div>
            </div>
            <div>
              <div className="flex justify-between text-xs mb-1">
                <span className="text-slate-400">Disk</span>
                <span className="text-white font-mono">{sys.disk?.used_gb} / {sys.disk?.total_gb} GB</span>
              </div>
              <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                <div className="h-full bg-neon-cyan transition-all" style={{ width: `${sys.disk?.percent}%` }} />
              </div>
            </div>
          </div>
        </Card>

        <Card>
          <h3 className="text-sm font-bold text-white mb-4 flex items-center gap-2">
            <MemoryStick size={16} className="text-neon-green" /> GPU Usage
          </h3>
          {gpu?.devices?.length ? (
            gpu.devices.map((dev: any, i: number) => (
              <div key={i} className="mb-3">
                <div className="flex justify-between text-xs mb-1">
                  <span className="text-slate-400">{dev.name}</span>
                  <span className="text-white font-mono">{dev.utilization_percent}%</span>
                </div>
                <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                  <div className="h-full bg-neon-green transition-all" style={{ width: `${dev.utilization_percent}%` }} />
                </div>
                <div className="text-[10px] text-slate-500 mt-1">
                  VRAM: {dev.memory_allocated_mb} / {dev.memory_total_mb} MB
                </div>
              </div>
            ))
          ) : (
            <p className="text-xs text-slate-400">No GPUs detected.</p>
          )}
        </Card>
      </div>

      {/* Job Distribution */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-4">Job Distribution</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {jobs.by_status ? (
            Object.entries(jobs.by_status).map(([status, count]: any) => (
              <div key={status} className="text-center p-4 bg-cyber-800 rounded-xl border border-white/5">
                <div className="text-2xl font-bold text-white">{count}</div>
                <div className="text-xs uppercase tracking-wider text-slate-400 mt-1">{status}</div>
              </div>
            ))
          ) : (
            <p className="text-slate-400 text-xs col-span-full">No job data available.</p>
          )}
        </div>
      </Card>

      {/* Quick Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Total Jobs</div>
          <div className="text-2xl font-bold text-white mt-1">{jobs.total || 0}</div>
        </Card>
        <Card className="text-center">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Avg Duration</div>
          <div className="text-2xl font-bold text-white mt-1">
            {jobs.average_completion_seconds ? `${Math.round(jobs.average_completion_seconds)}s` : '—'}
          </div>
        </Card>
        <Card className="text-center">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Uptime</div>
          <div className="text-2xl font-bold text-white mt-1">{sys?.uptime_human || '—'}</div>
        </Card>
        <Card className="text-center">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Python</div>
          <div className="text-2xl font-bold text-white mt-1">{sys?.python_version || '—'}</div>
        </Card>
      </div>
    </div>
  )
}