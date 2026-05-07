import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { ResearcherNote } from '../types'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'

export default function NoteDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [isEditing, setIsEditing] = useState(false)
  const [editForm, setEditForm] = useState({ title: '', content: '' })

  const { data: note, isLoading } = useQuery<ResearcherNote>({
    queryKey: ['note', id],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.DETAIL(id!))).data,
    enabled: !!id,
  })

  const updateMutation = useMutation({
    mutationFn: async (data: Partial<ResearcherNote>) => api.patch(ENDPOINTS.NOTES.DETAIL(id!), data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['note', id] })
      setIsEditing(false)
    },
  })

  if (isLoading || !note) return <Spinner />

  const handleEdit = () => {
    setEditForm({ title: note.title, content: note.content })
    setIsEditing(true)
  }

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl mx-auto">
      <div className="flex justify-between items-center">
        <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>← Back</Button>
        <div className="flex gap-2">
          <Button variant="secondary" size="sm" onClick={handleEdit}>Edit</Button>
          <Button variant="danger" size="sm" onClick={() => { /* delete logic */ }}>Delete</Button>
        </div>
      </div>

      <Card>
        {isEditing ? (
          <div className="space-y-4">
            <Input
              label="Title"
              value={editForm.title}
              onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
            />
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white min-h-[200px] focus:outline-none focus:border-neon-blue/50"
              value={editForm.content}
              onChange={(e) => setEditForm({ ...editForm, content: e.target.value })}
            />
            <div className="flex gap-2">
              <Button size="sm" onClick={() => updateMutation.mutate({ title: editForm.title, content: editForm.content })}>
                Save
              </Button>
              <Button variant="ghost" size="sm" onClick={() => setIsEditing(false)}>Cancel</Button>
            </div>
          </div>
        ) : (
          <>
            <div className="flex items-start justify-between mb-4">
              <h1 className="text-2xl font-black text-white">{note.title || 'Untitled'}</h1>
              <div className="flex gap-2">
                {note.tags.map((tag) => (
                  <Badge key={tag} status={tag}>{tag}</Badge>
                ))}
              </div>
            </div>
            <div className="prose prose-invert text-slate-300 whitespace-pre-wrap">
              {note.content}
            </div>
            <div className="mt-4 text-xs text-slate-500">
              by {note.author_username} · Updated {new Date(note.updated_at).toLocaleString()}
            </div>
          </>
        )}
      </Card>
    </div>
  )
}