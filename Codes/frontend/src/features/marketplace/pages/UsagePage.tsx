import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'

export default function UsagePage() {
  const { data } = useQuery({
    queryKey: ['marketplace-usage'],
    queryFn: () => api.get('/marketplace/usage/').then(r => r.data),
  })

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Usage</h1>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Total GPU Hours</p>
          <p className="text-2xl font-bold">{data?.total_hours || 0}</p>
        </div>
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Total Cost</p>
          <p className="text-2xl font-bold">${data?.total_cost || '0.00'}</p>
        </div>
      </div>
      <div className="space-y-2">
        {data?.records?.map((r: any) => (
          <div key={r.job_id} className="glass p-3 flex justify-between">
            <span className="font-mono text-blue-400">{r.job_id}</span>
            <span>{r.gpu_model} · {r.hours}h · ${r.cost}</span>
          </div>
        ))}
      </div>
    </div>
  )
}