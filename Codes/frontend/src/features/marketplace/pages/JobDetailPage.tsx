import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import api from '../../../api/client'

export default function JobDetailPage() {
  const { jobId } = useParams()
  const { data, isLoading } = useQuery({
    queryKey: ['job', jobId],
    queryFn: () => api.get(`/marketplace/jobs/${jobId}/`).then(r => r.data),
    enabled: !!jobId,
  })

  if (isLoading) return <div className="p-6">Loading...</div>
  if (!data) return <div className="p-6">Job not found</div>

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Job {data.id.slice(0, 8)}</h1>
      <div className="glass p-4 space-y-2">
        <p>Status: <span className="font-bold">{data.status}</span></p>
        <p>GPU: {data.listing_gpu_model}</p>
        <p>Price: ${data.price_per_hour}/hr</p>
        <p>Duration: {data.hours_consumed}h</p>
        <p>Total cost: ${data.total_cost_usd}</p>
        <p>Started: {data.started_at ? new Date(data.started_at).toLocaleString() : 'N/A'}</p>
        <p>Completed: {data.completed_at ? new Date(data.completed_at).toLocaleString() : 'N/A'}</p>
      </div>
      <div className="glass p-4 mt-4">
        <h2 className="font-bold mb-2">Events</h2>
        <div className="space-y-1 max-h-64 overflow-y-auto">
          {data.events?.map((e: any, i: number) => (
            <div key={i} className="text-sm border-b border-white/5 py-1">
              <span className="text-slate-400">{new Date(e.created_at).toLocaleTimeString()}</span>
              <span className="ml-2 font-medium">{e.type}</span>
              <span className="ml-2 text-slate-300">{e.description}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}