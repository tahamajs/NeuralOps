import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function StatisticsPage() {
  // This endpoint is from web_dashboard's statistics view. Ensure it's mounted at /api/statistics/ or use the Django view that returns context.
  // Actually, the web_dashboard views `statistics` returns a template context, not JSON.
  // We need to call the REST API /api/jobs/stats/ and maybe additional custom endpoints.
  // We'll use /api/jobs/stats/ for job stats and /api/experiments/ for runs.
  const { data: jobStats } = useQuery({ queryKey: ['jobStats'], queryFn: async () => (await api.get('/api/jobs/stats/')).data })
  const { data: experiments = [] } = useQuery({ queryKey: ['experiments'], queryFn: async () => (await api.get('/api/experiments/')).data })

  const completed = experiments.filter((e: any) => e.status === 'completed').length
  const failed = experiments.filter((e: any) => e.status === 'failed').length
  const running = experiments.filter((e: any) => e.status === 'running').length

  if (!jobStats) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        STATISTICS <span className="text-neon-blue">OVERVIEW</span>
      </h1>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Total Jobs</div>
          <div className="text-2xl font-bold text-white">{jobStats.total}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Completed</div>
          <div className="text-2xl font-bold text-green-400">{completed}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Running</div>
          <div className="text-2xl font-bold text-blue-400">{running}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Failed</div>
          <div className="text-2xl font-bold text-red-400">{failed}</div>
        </Card>
      </div>
      {jobStats.average_completion_seconds && (
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Avg Completion Time</div>
          <div className="text-xl font-bold text-white">{jobStats.average_completion_seconds} sec</div>
        </Card>
      )}
    </div>
  )
}