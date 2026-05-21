import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import api from '../../../api/client'

export default function MyJobsPage() {
  const { data } = useQuery({
    queryKey: ['my-jobs'],
    queryFn: () => api.get('/marketplace/jobs/').then(r => r.data),
  })
  const nav = useNavigate()

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">My Compute Jobs</h1>
      <div className="space-y-2">
        {data?.map((job: any) => (
          <div
            key={job.id}
            onClick={() => nav(`/marketplace/jobs/${job.id}`)}
            className="glass p-4 cursor-pointer hover:bg-white/10 transition-colors"
          >
            <div className="flex justify-between">
              <span className="font-mono text-blue-400">{job.id.slice(0, 8)}</span>
              <span className={`px-2 py-0.5 rounded-full text-xs font-bold ${
                job.status === 'running' ? 'bg-blue-500/20 text-blue-300' :
                job.status === 'completed' ? 'bg-green-500/20 text-green-300' :
                job.status === 'failed' ? 'bg-red-500/20 text-red-300' : 'bg-slate-500/20 text-slate-300'
              }`}>{job.status}</span>
            </div>
            <p className="text-sm">{job.listing_gpu_model} · ${job.price_per_hour}/hr · {job.hours_consumed}h · ${job.total_cost_usd}</p>
          </div>
        ))}
        {data?.length === 0 && <p className="text-slate-500">No jobs yet.</p>}
      </div>
    </div>
  )
}