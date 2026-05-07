import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { ResearcherNote } from '../types'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { EmptyState } from '../../../components/ui/EmptyState'
import { formatDistanceToNow } from 'date-fns' // you can use date‑fns or a custom formatter

export default function NotesListPage() {
  const [search, setSearch] = useState('')
  const queryClient = useQueryClient()

  const { data: notes = [], isLoading } = useQuery<ResearcherNote[]>({
    queryKey: ['notes'],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.LIST)).data.results || [],
  })

  const deleteMutation = useMutation({
    mutationFn: async (id: string) => api.delete(ENDPOINTS.NOTES.DETAIL(id)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['notes'] }),
  })

  const filtered = notes.filter(
    (n) => n.title.toLowerCase().includes(search.toLowerCase()) || n.content.toLowerCase().includes(search.toLowerCase())
  )

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-black text-white tracking-tighter">
          RESEARCH <span className="text-neon-blue">NOTES</span>
        </h1>
        <Link to="/notes/new">
          <Button size="sm">+ New Note</Button>
        </Link>
      </div>

      <Input
        placeholder="Search notes..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        className="max-w-md"
      />

      {filtered.length === 0 ? (
        <EmptyState message="No notes found" />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((note) => (
            <Card key={note.id} className="flex flex-col">
              <div className="flex items-start justify-between mb-2">
                <Link to={`/notes/${note.id}`} className="text-white font-semibold hover:text-neon-blue transition-colors">
                  {note.title || 'Untitled'}
                </Link>
                {note.pinned && <span className="text-yellow-400 text-xs">📌</span>}
              </div>
              <p className="text-xs text-slate-400 line-clamp-3 mb-3 flex-1">{note.content}</p>
              <div className="flex justify-between text-[10px] text-slate-500 uppercase tracking-wide">
                <span>{note.author_username}</span>
                <span>{formatDistanceToNow(new Date(note.created_at))} ago</span>
              </div>
              <div className="flex gap-2 mt-2">
                {note.tags.slice(0, 3).map((tag) => (
                  <Badge key={tag} status={tag}>{tag}</Badge>
                ))}
              </div>
              <Button variant="ghost" size="sm" className="mt-2 self-end" onClick={() => deleteMutation.mutate(note.id)}>
                Delete
              </Button>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}