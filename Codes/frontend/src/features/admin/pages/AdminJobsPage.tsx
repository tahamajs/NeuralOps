import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Input } from '../../../components/ui/Input'

export default function AdminJobsPage() {
  const queryClient = useQueryClient()
  const [selected, setSelected] = useState<string[]>([])
  const [filter, setFilter] = useState('')

  const { data: jobs = [], isLoading } = useQuery({
    queryKey: ['jobs'],
    queryFn: async () => (await api.get('/api/jobs/')).data,
    refetchInterval: 5000,
  })

  const deleteMutation = useMutation({
    mutationFn: async (ids: string[]) => api.post('/api/jobs/bulk-delete/', { job_ids: ids }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['jobs'] })
      setSelected([])
    },
  })

  const stopAllMutation = useMutation({
    mutationFn: async () => api.post('/api/runs/stop_all/'),
    onSuccess: () => alert('All jobs stopped'),
  })

  const filtered = jobs.filter((j: any) =>
    (j.name || '').toLowerCase().includes(filter.toLowerCase()) ||
    j.id.toLowerCase().includes(filter.toLowerCase())
  )

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-display font-bold text-white">Jobs Admin</h1>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => setSelected(filtered.map((j: any) => j.id))}>Select All</Button>
          <Button variant="danger" size="sm" disabled={selected.length === 0} onClick={() => deleteMutation.mutate(selected)}>
            Delete Selected ({selected.length})
          </Button>
          <Button variant="secondary" size="sm" onClick={() => stopAllMutation.mutate()}>Stop All</Button>
        </div>
      </div>
      <Input placeholder="Search jobs..." value={filter} onChange={e => setFilter(e.target.value)} />
      <Card>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-white/5">
                <th className="py-2 px-2">Select</th>
                <th className="py-2 px-2">ID</th>
                <th className="py-2 px-2">Name</th>
                <th className="py-2 px-2">Status</th>
                <th className="py-2 px-2">Priority</th>
                <th className="py-2 px-2">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filtered.map((job: any) => (
                <tr key={job.id}>
                  <td className="py-2 px-2">
                    <input type="checkbox" checked={selected.includes(job.id)} onChange={() => setSelected(prev => prev.includes(job.id) ? prev.filter(id => id !== job.id) : [...prev, job.id])} />
                  </td>
                  <td className="py-2 px-2 font-mono text-neon-blue">{job.id.slice(0,8)}</td>
                  <td className="py-2 px-2 text-white">{job.name || '—'}</td>
                  <td className="py-2 px-2"><Badge status={job.status}>{job.status}</Badge></td>
                  <td className="py-2 px-2">{job.priority}</td>
                  <td className="py-2 px-2 flex gap-1">
                    <Button variant="ghost" size="sm" onClick={() => window.open(`/experiments/${job.id}`, '_blank')}>Detail</Button>
                    {job.status === 'running' && <Button variant="ghost" size="sm" onClick={() => api.post(`/api/jobs/${job.id}/stop/`)}>Stop</Button>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}