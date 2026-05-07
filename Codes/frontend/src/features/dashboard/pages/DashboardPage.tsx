import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Link } from 'react-router-dom'
import { Experiment } from '../../../types/experiment'
import { useNavigate } from 'react-router-dom'
import { BarChart3, TrendingUp, Cpu, Activity, Plus } from 'lucide-react'

export default function DashboardPage() {
  const navigate = useNavigate()
  const { data: experiments = [], isLoading } = useQuery<Experiment[]>({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.LIST)).data,
    refetchInterval: 5000,
  })

  const { data: system } = useQuery({
    queryKey: ['systemStatus'],
    queryFn: async () => (await api.get('/api/system/status/')).data,
    refetchInterval: 10000,
  })

  const completed = experiments.filter(e => e.status === 'completed').length
  const running = experiments.filter(e => e.status === 'running').length
  const failed = experiments.filter(e => e.status === 'failed').length
  const avgAccuracy = completed > 0
    ? (experiments.filter(e => e.status === 'completed').reduce((sum, e) => sum + (e.final_metrics?.eval_accuracy || 0), 0) / completed).toFixed(4)
    : '—'

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Welcome Banner */}
      <div className="glass p-6 rounded-2xl bg-gradient-to-r from-neon-blue/10 to-purple-500/10 border border-white/10">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-display font-bold text-white">
              Welcome back, <span className="text-neon-blue">Researcher</span>
            </h1>
            <p className="text-slate-400 mt-1 text-sm">Here's what's happening across your experiments.</p>
          </div>
          <Button onClick={() => navigate('/launcher')} className="flex items-center gap-2">
            <Plus size={16} /> New Experiment
          </Button>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Total Experiments</p>
              <p className="text-3xl font-display font-bold text-white mt-1">{experiments.length}</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-neon-blue/10 flex items-center justify-center">
              <BarChart3 size={20} className="text-neon-blue" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Running Jobs</p>
              <p className="text-3xl font-display font-bold text-neon-orange mt-1">{running}</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-neon-orange/10 flex items-center justify-center">
              <Activity size={20} className="text-neon-orange" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Avg Accuracy</p>
              <p className="text-3xl font-display font-bold text-neon-green mt-1">{avgAccuracy}</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-neon-green/10 flex items-center justify-center">
              <TrendingUp size={20} className="text-neon-green" />
            </div>
          </div>
        </Card>
        <Card className="card-hover-effect">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">CPU Usage</p>
              <p className="text-3xl font-display font-bold text-neon-purple mt-1">{system?.cpu?.percent || 0}%</p>
            </div>
            <div className="w-10 h-10 rounded-xl bg-neon-purple/10 flex items-center justify-center">
              <Cpu size={20} className="text-neon-purple" />
            </div>
          </div>
        </Card>
      </div>

      {/* Recent Experiments Table */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-display font-bold text-white">Recent Experiments</h2>
          <Link to="/experiments" className="text-neon-blue text-sm hover:underline">View all →</Link>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-slate-400 border-b border-white/5">
                <th className="py-3 px-4 font-medium">Name</th>
                <th className="py-3 px-4 font-medium">Task</th>
                <th className="py-3 px-4 font-medium">Status</th>
                <th className="py-3 px-4 font-medium">Accuracy</th>
                <th className="py-3 px-4 font-medium">Created</th>
                <th className="py-3 px-4 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {experiments.slice(0, 5).map((exp) => (
                <tr key={exp.id} className="hover:bg-white/5 transition-colors cursor-pointer" onClick={() => navigate(`/experiments/${exp.id}`)}>
                  <td className="py-3 px-4 text-white">{exp.name || exp.id.slice(0,8)}</td>
                  <td className="py-3 px-4 uppercase text-xs">{exp.config?.task_name || '—'}</td>
                  <td className="py-3 px-4"><Badge status={exp.status}>{exp.status}</Badge></td>
                  <td className="py-3 px-4">{exp.final_metrics?.eval_accuracy?.toFixed(4) || '—'}</td>
                  <td className="py-3 px-4 text-slate-400">{new Date(exp.created_at).toLocaleDateString()}</td>
                  <td className="py-3 px-4 text-right">
                    <Button variant="ghost" size="sm" onClick={(e) => { e.stopPropagation(); navigate(`/experiments/${exp.id}`) }}>View</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Quick Actions */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Link to="/analysis" className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3">
          <BarChart3 size={24} className="text-neon-blue" />
          <span className="text-sm font-medium text-white">Analysis</span>
        </Link>
        <Link to="/notes/notebook" className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3">
          <BookOpen size={24} className="text-neon-purple" />
          <span className="text-sm font-medium text-white">Notebook</span>
        </Link>
        <Link to="/paper/new" className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3">
          <FileText size={24} className="text-neon-pink" />
          <span className="text-sm font-medium text-white">Papers</span>
        </Link>
        <Link to="/cluster" className="glass p-4 rounded-xl hover:bg-white/10 transition-all flex items-center gap-3">
          <Cpu size={24} className="text-neon-green" />
          <span className="text-sm font-medium text-white">Cluster</span>
        </Link>
      </div>
    </div>
  )
}