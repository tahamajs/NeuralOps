import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { BarChart3, Clock, Cpu, Layers, TrendingUp } from 'lucide-react' // Added TrendingUp

interface JobStats {
  total_jobs: number
  average_completion_seconds: number | null
  // ... add other fields as needed
}

export default function StatisticsPage() {
  const { data: jobStats, isLoading: jobsLoading, error: jobsError } = useQuery<JobStats>({
    queryKey: ['jobStats'],
    queryFn: async () => (await api.get('/jobs/stats/')).data,
  })

  const { data: experiments = [], isLoading: expsLoading, error: expsError } = useQuery({
    queryKey: ['experiments'],
    queryFn: async () => (await api.get('/experiments/')).data,
  })

  const isLoading = jobsLoading || expsLoading
  if (isLoading) return <Spinner />

  if (jobsError || expsError) {
    return (
      <Card className="p-6 text-center">
        <p className="text-red-400">Failed to load statistics. Please refresh.</p>
      </Card>
    )
  }

  const completed = experiments.filter((e: any) => e.status === 'completed')
  const running = experiments.filter((e: any) => e.status === 'running')
  const failed = experiments.filter((e: any) => e.status === 'failed')

  const avgAccuracy =
    completed.length > 0
      ? completed.reduce((sum: number, e: any) => sum + (e.final_metrics?.eval_accuracy || 0), 0) /
        completed.length
      : null

  const avgDuration = jobStats?.average_completion_seconds || null

  // Model usage
  const modelCounts = experiments.reduce((acc: Record<string, number>, e: any) => {
    const model = e.config?.model_name || 'unknown'
    acc[model] = (acc[model] || 0) + 1
    return acc
  }, {})

  const topModels = Object.entries(modelCounts)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 5)

  // Task usage
  const taskCounts = experiments.reduce((acc: Record<string, number>, e: any) => {
    const task = e.config?.task_name || 'unknown'
    acc[task] = (acc[task] || 0) + 1
    return acc
  }, {})

  const topTasks = Object.entries(taskCounts)
    .sort(([, a], [, b]) => (b as number) - (a as number))
    .slice(0, 5)

  const gpuJobs = experiments.filter((e: any) => (e.config?.gpu_requirements || 0) > 0).length

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex items-center gap-3">
        <BarChart3 size={24} className="text-neon-blue" />
        <h1 className="text-2xl font-display font-bold text-white">Global Statistics</h1>
      </div>

      {/* KPI Row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center p-5">
          <div className="text-3xl font-bold text-white">{experiments.length}</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-2">Total Runs</div>
        </Card>
        <Card className="text-center p-5">
          <div className="text-3xl font-bold text-green-400">{completed.length}</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-2">Completed</div>
        </Card>
        <Card className="text-center p-5">
          <div className="text-3xl font-bold text-blue-400">{running.length}</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-2">Running</div>
        </Card>
        <Card className="text-center p-5">
          <div className="text-3xl font-bold text-red-400">{failed.length}</div>
          <div className="text-xs text-slate-400 uppercase tracking-wider mt-2">Failed</div>
        </Card>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {avgAccuracy !== null && (
          <Card className="flex items-center gap-3 p-4">
            <div className="w-10 h-10 rounded-xl bg-green-500/10 flex items-center justify-center">
              <TrendingUp size={20} className="text-green-400" />
            </div>
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Avg Accuracy</p>
              <p className="text-xl font-bold text-white">{avgAccuracy.toFixed(4)}</p>
            </div>
          </Card>
        )}
        {avgDuration !== null && (
          <Card className="flex items-center gap-3 p-4">
            <div className="w-10 h-10 rounded-xl bg-neon-blue/10 flex items-center justify-center">
              <Clock size={20} className="text-neon-blue" />
            </div>
            <div>
              <p className="text-xs text-slate-400 uppercase tracking-wider">Avg Duration</p>
              <p className="text-xl font-bold text-white">{Math.round(avgDuration)}s</p>
            </div>
          </Card>
        )}
        <Card className="flex items-center gap-3 p-4">
          <div className="w-10 h-10 rounded-xl bg-purple-500/10 flex items-center justify-center">
            <Cpu size={20} className="text-purple-400" />
          </div>
          <div>
            <p className="text-xs text-slate-400 uppercase tracking-wider">GPU Jobs</p>
            <p className="text-xl font-bold text-white">{gpuJobs}</p>
          </div>
        </Card>
      </div>

      {/* Model & Task Usage */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Top Models</h3>
          {topModels.length ? (
            topModels.map(([model, count]) => (
              <div key={model} className="flex justify-between text-xs py-1.5 border-b border-white/5">
                <span className="text-slate-400">{model}</span>
                <span className="text-white font-mono">{count}</span>
              </div>
            ))
          ) : (
            <p className="text-slate-500 text-sm">No models used yet.</p>
          )}
        </Card>
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Top Tasks</h3>
          {topTasks.length ? (
            topTasks.map(([task, count]) => (
              <div key={task} className="flex justify-between text-xs py-1.5 border-b border-white/5">
                <span className="text-slate-400 uppercase">{task}</span>
                <span className="text-white font-mono">{count}</span>
              </div>
            ))
          ) : (
            <p className="text-slate-500 text-sm">No tasks yet.</p>
          )}
        </Card>
      </div>
    </div>
  )
}