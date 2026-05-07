import { useParams, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'
import { ArrowLeft, Check } from 'lucide-react'

export default function NoteMentionsPage() {
  const { noteId } = useParams<{ noteId: string }>()
  const queryClient = useQueryClient()

  const { data: mentions = [], isLoading } = useQuery({
    queryKey: ['noteMentions', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/mentions/`)).data,
    enabled: !!noteId,
  })

  const markReadMut = useMutation({
    mutationFn: (mentionId: string) => api.post(`/api/notes/v1/mentions/${mentionId}/read/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['noteMentions', noteId] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex items-center gap-2">
        <Link to={`/notes/${noteId}`} className="text-slate-400 hover:text-white">
          <ArrowLeft size={18} />
        </Link>
        <h1 className="text-2xl font-display font-bold text-white">Mentions</h1>
      </div>

      {mentions.length === 0 ? (
        <EmptyState message="No mentions in this note." />
      ) : (
        <div className="space-y-2">
          {mentions.map((m: any) => (
            <Card key={m.user} className="flex items-center justify-between">
              <div>
                <span className="text-sm font-semibold text-white">{m.user}</span>
                <p className="text-xs text-slate-400">{new Date(m.created_at).toLocaleString()}</p>
              </div>
              {!m.is_read && (
                <Button size="sm" variant="ghost" onClick={() => markReadMut.mutate(m.id)}>
                  <Check size={14} className="mr-1" /> Mark Read
                </Button>
              )}
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}