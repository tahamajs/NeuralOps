import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'

export default function AdminDashboard() {
  const { data: health } = useQuery({ queryKey: ['health'], queryFn: async () => (await api.get('/api/health/')).data })
  const { data: version } = useQuery({ queryKey: ['version'], queryFn: async () => (await api.get('/api/version/')).data })
  const { data: jobStats } = useQuery({ queryKey: ['jobStats'], queryFn: async () => (await api.get('/api/jobs/stats/')).data })

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white">Admin Dashboard</h1>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        <Card><div className="text-xs text-slate-400 uppercase">System Status</div><div className="text-xl font-bold text-white">{health?.status || 'Unknown'}</div></Card>
        <Card><div className="text-xs text-slate-400 uppercase">API Version</div><div className="text-xl font-bold text-white">{version?.api_version || '—'}</div></Card>
        <Card><div className="text-xs text-slate-400 uppercase">Total Jobs</div><div className="text-xl font-bold text-white">{jobStats?.total || 0}</div></Card>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Link to="/admin/jobs"><Card hover><p className="font-bold text-white">Jobs Management</p><p className="text-xs text-slate-400">Bulk delete, stop all</p></Card></Link>
        <Link to="/admin/model-versions"><Card hover><p className="font-bold text-white">Model Versions</p><p className="text-xs text-slate-400">Publish, archive</p></Card></Link>
        <Link to="/cluster"><Card hover><p className="font-bold text-white">Cluster Status</p><p className="text-xs text-slate-400">GPU nodes & queue</p></Card></Link>
        <a href="/admin/" target="_blank"><Card hover><p className="font-bold text-white">Django Admin</p><p className="text-xs text-slate-400">Full database access</p></Card></a>
      </div>
      {jobStats && (
        <Card>
          <h2 className="text-sm font-bold text-white mb-2">Job Breakdown</h2>
          <div className="grid grid-cols-3 gap-2 text-center text-xs">
            {Object.entries(jobStats.by_status).map(([status, count]: any) => (
              <div key={status} className="bg-cyber-800 p-2 rounded">
                <div className="font-bold text-white">{count}</div>
                <div className="text-slate-500 capitalize">{status}</div>
              </div>
            ))}
          </div>
        </Card>
      )}
    </div>
  )
}