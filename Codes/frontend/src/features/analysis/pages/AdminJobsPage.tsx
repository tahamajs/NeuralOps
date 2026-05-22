import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Spinner, Table, Badge } from '../../../components/ui'
import { Trash2, StopCircle } from 'lucide-react'

export default function AdminJobsPage() {
  const queryClient = useQueryClient()
  const { data, isLoading } = useQuery({
    queryKey: ['admin-jobs'],
    queryFn: async () => (await api.get('/jobs/?all=true')).data,
  })

  const stopJob = useMutation({
    mutationFn: (jobId: string) => api.post(`/jobs/${jobId}/stop/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['admin-jobs'] }),
  })

  const deleteJob = useMutation({
    mutationFn: (jobId: string) => api.delete(`/jobs/${jobId}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['admin-jobs'] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-white">Manage Jobs</h1>
      <Table>
        <thead>
          <tr><th>ID</th><th>Status</th><th>User</th><th>Created</th><th>Actions</th></tr>
        </thead>
        <tbody>
          {(data || []).map((job: any) => (
            <tr key={job.id}>
              <td className="font-mono">{job.id.slice(0,8)}</td>
              <td><Badge variant={job.status}>{job.status}</Badge></td>
              <td>{job.user__username || 'system'}</td>
              <td>{new Date(job.created_at).toLocaleString()}</td>
              <td className="space-x-2">
                <Button size="sm" variant="ghost" onClick={() => stopJob.mutate(job.id)}>
                  <StopCircle size={14} />
                </Button>
                <Button size="sm" variant="ghost" onClick={() => deleteJob.mutate(job.id)}>
                  <Trash2 size={14} className="text-red-400" />
                </Button>
              </td>
            </tr>
          ))}
        </tbody>
      </Table>
    </div>
  )
}