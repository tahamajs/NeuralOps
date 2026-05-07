import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'

export default function TemplateDetailPage() {
  const { templateId } = useParams<{ templateId: string }>()
  const queryClient = useQueryClient()
  const [editForm, setEditForm] = useState<Record<string, any>>({})

  const { data: template, isLoading } = useQuery({
    queryKey: ['template', templateId],
    queryFn: async () => (await api.get(`/api/experiments/templates/${templateId}/`)).data,
    onSuccess: (data) => setEditForm(data),
  })

  const updateMutation = useMutation({
    mutationFn: async (form: any) => api.patch(`/api/experiments/templates/${templateId}/`, form),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['template', templateId] }),
  })

  if (isLoading || !template) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white">{template.name}</h1>
      <Card>
        <div className="space-y-4">
          <Input label="Name" value={editForm.name} onChange={(e) => setEditForm({ ...editForm, name: e.target.value })} />
          <Input label="Description" value={editForm.description} onChange={(e) => setEditForm({ ...editForm, description: e.target.value })} />
          <Input label="Category" value={editForm.category} onChange={(e) => setEditForm({ ...editForm, category: e.target.value })} />
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest">Config (JSON)</label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white font-mono text-xs h-64 mt-1"
              value={JSON.stringify(editForm.config || {}, null, 2)}
              onChange={(e) => { try { setEditForm({ ...editForm, config: JSON.parse(e.target.value) }) } catch {} }}
            />
          </div>
          <Button onClick={() => updateMutation.mutate(editForm)} disabled={updateMutation.isPending}>
            Save Changes
          </Button>
        </div>
      </Card>
    </div>
  )
}