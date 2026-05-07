import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'

export default function NoteMentionsPage() {
  const { noteId } = useParams<{ noteId: string }>()
  const queryClient = useQueryClient()
  const { data: mentions = [], isLoading } = useQuery({
    queryKey: ['noteMentions', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/mentions/`)).data,
  })

  const markReadMut = useMutation({
    mutationFn: async (mentionId: string) => api.post(`/api/notes/v1/mentions/${mentionId}/read/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['noteMentions', noteId] }),
  })

  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">MENTIONS</h1>
      {mentions.map((m: any) => (
        <Card key={m.user} className="flex justify-between items-center">
          <div>
            <span className="text-white">{m.user}</span>
            <p className="text-xs text-slate-400">{new Date(m.created_at).toLocaleString()}</p>
          </div>
          {!m.is_read && (
            <Button size="sm" variant="ghost" onClick={() => markReadMut.mutate(m.id)}>
              Mark Read
            </Button>
          )}
        </Card>
      ))}
    </div>
  )
}