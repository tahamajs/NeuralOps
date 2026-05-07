import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import { ResearcherNote } from '../types'

export default function NoteDetailPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()

  const [isEditing, setIsEditing] = useState(false)
  const [editForm, setEditForm] = useState({ title: '', content: '', tags: '' })
  const [shareUsername, setShareUsername] = useState('')
  const [showSharePanel, setShowSharePanel] = useState(false)
  const [activeTab, setActiveTab] = useState<'view' | 'versions' | 'replies' | 'mentions'>('view')

  const { data: note, isLoading } = useQuery<ResearcherNote>({
    queryKey: ['note', id],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.DETAIL(id!))).data,
    enabled: !!id,
  })

  // Mutations
  const updateMut = useMutation({
    mutationFn: async (data: Partial<ResearcherNote>) => {
      const tagsArray = editForm.tags.split(',').map((t) => t.trim()).filter(Boolean)
      return api.patch(ENDPOINTS.NOTES.DETAIL(id!), { ...data, tags: tagsArray })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['note', id] })
      setIsEditing(false)
    },
  })

  const deleteMut = useMutation({
    mutationFn: () => api.delete(ENDPOINTS.NOTES.DETAIL(id!)),
    onSuccess: () => navigate('/notes'),
  })

  const shareMut = useMutation({
    mutationFn: async (username: string) =>
      api.post(`/api/notes/v1/notes/${id}/share/`, { username }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['note', id] }),
  })

  const reactMut = useMutation({
    mutationFn: async (reaction: string) =>
      api.post(ENDPOINTS.NOTES.REACT(id!), { reaction }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['note', id] }),
  })

  const pinMut = useMutation({
    mutationFn: async (pinned: boolean) =>
      api.patch(ENDPOINTS.NOTES.DETAIL(id!), { pinned }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['note', id] }),
  })

  if (isLoading || !note) return <Spinner />

  const handleEdit = () => {
    setEditForm({
      title: note.title,
      content: note.content,
      tags: (note.tags || []).join(', '),
    })
    setIsEditing(true)
    setActiveTab('view')
  }

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      {/* ── Top Bar ── */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}>← Back</Button>
          {note.pinned && <span className="text-yellow-400">📌 Pinned</span>}
          {note.is_public && <Badge status="public">Public</Badge>}
        </div>
        <div className="flex gap-2 flex-wrap">
          <Button variant="ghost" size="sm" onClick={() => pinMut.mutate(!note.pinned)}>
            {note.pinned ? 'Unpin' : 'Pin'}
          </Button>
          <Button variant="ghost" size="sm" onClick={handleEdit}>Edit</Button>
          <Button variant="ghost" size="sm" onClick={() => setShowSharePanel(!showSharePanel)}>
            👥 Share
          </Button>
          <Button variant="ghost" size="sm" onClick={() => navigate(`/notes/${id}/versions`)}>
            Versions ({note.version})
          </Button>
          <Button variant="danger" size="sm" onClick={() => deleteMut.mutate()}>Delete</Button>
        </div>
      </div>

      {/* ── Share Panel ── */}
      {showSharePanel && (
        <Card className="animate-slide-up">
          <h3 className="text-sm font-bold text-white mb-3">Share Note</h3>
          <div className="flex gap-2 mb-3">
            <Input
              placeholder="Username to share with..."
              value={shareUsername}
              onChange={(e) => setShareUsername(e.target.value)}
              className="flex-1"
            />
            <Button size="sm" onClick={() => { shareMut.mutate(shareUsername); setShareUsername('') }}>
              Share
            </Button>
          </div>
          {note.shared_with?.length > 0 && (
            <div>
              <p className="text-xs text-slate-500 mb-2">Shared with:</p>
              <div className="flex flex-wrap gap-2">
                {note.shared_with.map((u) => (
                  <span key={u} className="px-2 py-1 bg-cyber-800 rounded text-xs text-neon-blue">{u}</span>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* ── Content Area ── */}
      {!isEditing ? (
        <Card>
          <div className="flex items-start justify-between mb-4">
            <h1 className="text-2xl font-black text-white">{note.title || 'Untitled'}</h1>
            <span className="text-xs text-slate-500">v{note.version}</span>
          </div>
          <div className="flex flex-wrap gap-1 mb-4">
            {note.tags?.map((tag) => (
              <Badge key={tag} status={tag}>{tag}</Badge>
            ))}
          </div>
          <div className="prose prose-invert max-w-none text-slate-300 whitespace-pre-wrap leading-relaxed text-sm">
            {note.content}
          </div>
          <div className="mt-6 flex justify-between items-center text-xs text-slate-500 border-t border-white/5 pt-4">
            <span>by {note.author_username}</span>
            <span>Updated {new Date(note.updated_at).toLocaleString()}</span>
          </div>

          {/* ── Reactions ── */}
          <div className="mt-4 flex gap-2 border-t border-white/5 pt-4">
            {['👍', '❤️', '🔥', '💡', '❓'].map((emoji) => (
              <button
                key={emoji}
                onClick={() => reactMut.mutate(emoji)}
                className={`px-2 py-1 rounded-lg text-sm transition-all ${
                  (note.reactions?.[emoji] || []).length > 0
                    ? 'bg-neon-blue/20 border border-neon-blue/30'
                    : 'bg-white/5 border border-white/10 hover:bg-white/10'
                }`}
              >
                {emoji} {(note.reactions?.[emoji] || []).length}
              </button>
            ))}
          </div>
        </Card>
      ) : (
        <Card>
          <div className="space-y-4">
            <Input
              label="Title"
              value={editForm.title}
              onChange={(e) => setEditForm({ ...editForm, title: e.target.value })}
            />
            <div>
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Content</label>
              <textarea
                className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white text-sm min-h-[300px] focus:border-neon-blue/50 transition-colors scrollbar-thin"
                value={editForm.content}
                onChange={(e) => setEditForm({ ...editForm, content: e.target.value })}
                placeholder="Write your research notes here..."
              />
            </div>
            <Input
              label="Tags (comma separated)"
              value={editForm.tags}
              onChange={(e) => setEditForm({ ...editForm, tags: e.target.value })}
            />
            <div className="flex gap-2">
              <Button onClick={() => updateMut.mutate(editForm)} disabled={updateMut.isPending}>
                Save Changes
              </Button>
              <Button variant="ghost" onClick={() => setIsEditing(false)}>Cancel</Button>
            </div>
          </div>
        </Card>
      )}

      {/* ── Quick Links ── */}
      <div className="flex flex-wrap gap-3 text-xs">
        <Link to={`/notes/${id}/versions`} className="text-neon-blue hover:underline">Version History</Link>
        <Link to={`/notes/${id}/replies`} className="text-neon-blue hover:underline">Replies</Link>
        <Link to={`/notes/${id}/mentions`} className="text-neon-blue hover:underline">Mentions</Link>
        <button onClick={() => window.print()} className="text-neon-blue hover:underline">Print / PDF</button>
        <a href={ENDPOINTS.NOTES.EXPORT_PDF(id!)} className="text-neon-blue hover:underline">Export PDF</a>
      </div>
    </div>
  )
}