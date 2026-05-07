import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'

export default function WebhookManagerPage() {
  const queryClient = useQueryClient()
  const [form, setForm] = useState({ url: '', events: 'note.created,note.updated' })

  const { data: webhooks = [], isLoading } = useQuery({
    queryKey: ['webhooks'],
    queryFn: async () => (await api.get('/api/notes/v1/webhooks/')).data,
  })

  const createMutation = useMutation({
    mutationFn: async (data: any) => api.post('/api/notes/v1/webhooks/', {
      ...data,
      events: data.events.split(',').map((e: string) => e.trim()),
    }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['webhooks'] })
      setForm({ url: '', events: '' })
    },
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => api.delete(`/api/notes/v1/webhooks/${id}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['webhooks'] }),
  })

  const testMutation = useMutation({
    mutationFn: async (id: string) => api.post(`/api/notes/v1/webhooks/${id}/test/`),
    onSuccess: () => alert('Webhook test triggered'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        WEBHOOK <span className="text-neon-blue">MANAGER</span>
      </h1>
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Add Webhook</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <Input placeholder="URL" value={form.url} onChange={e => setForm({ ...form, url: e.target.value })} />
          <Input placeholder="Events (comma)" value={form.events} onChange={e => setForm({ ...form, events: e.target.value })} />
          <Button onClick={() => createMutation.mutate(form)} disabled={createMutation.isPending}>
            Add
          </Button>
        </div>
      </Card>
      {webhooks.map((w: any) => (
        <Card key={w.id} className="flex justify-between items-center">
          <div className="text-sm">
            <p className="text-white">{w.url}</p>
            <p className="text-xs text-slate-500">{w.events.join(', ')}</p>
          </div>
          <div className="flex gap-2">
            <Button size="sm" variant="ghost" onClick={() => testMutation.mutate(w.id)}>Test</Button>
            <Button size="sm" variant="danger" onClick={() => deleteMutation.mutate(w.id)}>Delete</Button>
          </div>
        </Card>
      ))}
    </div>
  )
}