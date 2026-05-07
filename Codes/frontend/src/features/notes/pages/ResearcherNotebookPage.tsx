import { useState, useEffect, useMemo } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { marked } from 'marked'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'
import { useAuthStore } from '../../auth/store/authStore'

// ---------- Types ----------
interface Note {
  id: string
  title: string
  content: string
  author_username: string
  run_id?: string
  tags: string[]
  is_public: boolean
  pinned: boolean
  reactions: Record<string, string[]>
  version: number
  parent_note_id?: string
  reply_count: number
  created_at: string
  updated_at: string
}

// ---------- Sub-components (inline for simplicity) ----------

const NoteCard = ({ note, onView, onDelete, onReact }: {
  note: Note
  onView: (id: string) => void
  onDelete: (id: string) => void
  onReact: (id: string, reaction: string) => void
}) => (
  <Card hover className="relative group">
    {note.pinned && <span className="absolute top-2 right-2 text-yellow-400 text-xs">📌</span>}
    <div className="flex justify-between items-start gap-3">
      <div>
        <h3 className="text-white font-semibold text-sm">
          {note.title || 'Untitled'}
          {note.pinned && <span className="ml-2 text-yellow-400 text-xs">Pinned</span>}
        </h3>
        <p className="text-[10px] text-slate-400 mt-0.5">
          {note.author_username} · {new Date(note.created_at).toLocaleString()}
        </p>
      </div>
      <div className="flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
        <Button variant="ghost" size="sm" onClick={() => onView(note.id)}>👁️</Button>
        <Button variant="ghost" size="sm" onClick={() => onDelete(note.id)}>🗑️</Button>
      </div>
    </div>
    <div
      className="mt-2 text-sm text-slate-300 line-clamp-3 prose prose-invert max-w-none"
      dangerouslySetInnerHTML={{ __html: marked.parse(note.content.substring(0, 300)) }}
    />
    <div className="flex flex-wrap gap-1 mt-2">
      {(note.tags || []).slice(0, 5).map(tag => (
        <span key={tag} className="px-1.5 py-0.5 rounded bg-white/5 text-[9px] text-slate-400 uppercase">{tag}</span>
      ))}
    </div>
    <div className="flex items-center gap-1 mt-2 text-xs">
      {Object.entries(note.reactions || {}).map(([emoji, users]) => (
        <button
          key={emoji}
          onClick={() => onReact(note.id, emoji)}
          className={`px-1.5 py-0.5 rounded hover:bg-white/10 transition-colors ${users.length > 0 ? 'text-yellow-400' : 'text-slate-500'}`}
        >
          {emoji} {users.length}
        </button>
      ))}
      <button
        onClick={() => onReact(note.id, '👍')}
        className="px-1.5 py-0.5 rounded hover:bg-white/10 text-slate-500 hover:text-neon-blue"
      >
        +👍
      </button>
      <button
        onClick={() => onReact(note.id, '💡')}
        className="px-1.5 py-0.5 rounded hover:bg-white/10 text-slate-500 hover:text-neon-blue"
      >
        +💡
      </button>
    </div>
    {note.run_id && (
      <div className="mt-1 text-[10px] text-slate-500">Run: {note.run_id.slice(0, 8)}</div>
    )}
    {note.reply_count > 0 && (
      <div className="mt-1 text-[10px] text-slate-500">💬 {note.reply_count} replies</div>
    )}
  </Card>
)

const NoteModal = ({ noteId, onClose }: { noteId: string | null; onClose: () => void }) => {
  if (!noteId) return null

  const queryClient = useQueryClient()
  const [activeTab, setActiveTab] = useState<'view' | 'edit' | 'replies' | 'versions'>('view')
  const [editForm, setEditForm] = useState({ title: '', content: '', tags: '', is_public: false, pinned: false })
  const [replyContent, setReplyContent] = useState('')

  const { data: note, isLoading } = useQuery<Note>({
    queryKey: ['noteDetail', noteId],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.DETAIL(noteId))).data,
  })

  const { data: replies = [] } = useQuery({
    queryKey: ['noteReplies', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/replies/`)).data,
    enabled: activeTab === 'replies',
  })

  const { data: versions = [] } = useQuery({
    queryKey: ['noteVersions', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/versions/`)).data,
    enabled: activeTab === 'versions',
  })

  useEffect(() => {
    if (note) {
      setEditForm({
        title: note.title || '',
        content: note.content || '',
        tags: (note.tags || []).join(', '),
        is_public: note.is_public,
        pinned: note.pinned,
      })
    }
  }, [note])

  const pinMut = useMutation({
    mutationFn: (pinned: boolean) => api.patch(ENDPOINTS.NOTES.DETAIL(noteId), { pinned }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['noteDetail', noteId] }),
  })

  const publicMut = useMutation({
    mutationFn: (is_public: boolean) => api.patch(ENDPOINTS.NOTES.DETAIL(noteId), { is_public }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['noteDetail', noteId] }),
  })

  const deleteMut = useMutation({
    mutationFn: () => api.delete(ENDPOINTS.NOTES.DETAIL(noteId)),
    onSuccess: () => { onClose(); queryClient.invalidateQueries({ queryKey: ['notes'] }) },
  })

  const saveEditMut = useMutation({
    mutationFn: () => {
      const tagsArray = editForm.tags.split(',').map(t => t.trim()).filter(Boolean)
      return api.patch(ENDPOINTS.NOTES.DETAIL(noteId), { ...editForm, tags: tagsArray })
    },
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['noteDetail', noteId] }),
  })

  const postReplyMut = useMutation({
    mutationFn: () => api.post(`/api/notes/v1/notes/${noteId}/replies/`, { content: replyContent }),
    onSuccess: () => {
      setReplyContent('')
      queryClient.invalidateQueries({ queryKey: ['noteReplies', noteId] })
    },
  })

  if (isLoading) return <Spinner className="py-8" />

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 animate-fade-in">
      <div className="glass w-full max-w-4xl max-h-[90vh] overflow-y-auto m-4 p-6 scrollbar-thin">
        {/* Header */}
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-xl font-black text-white">{note?.title || 'Note Detail'}</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>✕</Button>
        </div>

        {/* Tabs */}
        <div className="flex gap-1 mb-4 border-b border-white/5 pb-2">
          {['view', 'edit', 'replies', 'versions'].map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab as any)}
              className={`px-3 py-1 rounded-t text-xs font-bold uppercase tracking-wider transition-colors ${
                activeTab === tab ? 'bg-neon-blue/20 text-neon-blue border-b-2 border-neon-blue' : 'text-slate-400 hover:text-white'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>

        {/* Tab Content */}
        {activeTab === 'view' && note && (
          <div>
            <div className="prose prose-invert max-w-none text-sm" dangerouslySetInnerHTML={{ __html: marked.parse(note.content) }} />
            <div className="mt-4 text-xs text-slate-400 space-y-1">
              <p>Tags: {note.tags?.map(t => <Badge key={t}>{t}</Badge>)}</p>
              <p>Run: {note.run_id || '—'}</p>
              <p>Public: {note.is_public ? 'Yes' : 'No'} · Pinned: {note.pinned ? 'Yes' : 'No'}</p>
              <p>Version: {note.version} · Created: {new Date(note.created_at).toLocaleString()} · Updated: {new Date(note.updated_at).toLocaleString()}</p>
            </div>
            <div className="flex gap-2 mt-3">
              <Button size="sm" variant="secondary" onClick={() => pinMut.mutate(!note.pinned)}>
                {note.pinned ? 'Unpin' : '📌 Pin'}
              </Button>
              <Button size="sm" variant="ghost" onClick={() => publicMut.mutate(!note.is_public)}>
                {note.is_public ? 'Make Private' : '🌍 Make Public'}
              </Button>
              <Button size="sm" variant="danger" onClick={() => deleteMut.mutate()}>Delete</Button>
            </div>
          </div>
        )}

        {activeTab === 'edit' && (
          <div className="space-y-3">
            <Input label="Title" value={editForm.title} onChange={e => setEditForm({ ...editForm, title: e.target.value })} />
            <div>
              <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Content (Markdown)</label>
              <textarea
                className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[200px] focus:border-neon-blue/50"
                value={editForm.content}
                onChange={e => setEditForm({ ...editForm, content: e.target.value })}
              />
            </div>
            <Input label="Tags (comma separated)" value={editForm.tags} onChange={e => setEditForm({ ...editForm, tags: e.target.value })} />
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={editForm.is_public} onChange={e => setEditForm({ ...editForm, is_public: e.target.checked })} className="accent-neon-blue" />
              <span className="text-xs text-slate-400">Public</span>
            </label>
            <label className="flex items-center gap-2">
              <input type="checkbox" checked={editForm.pinned} onChange={e => setEditForm({ ...editForm, pinned: e.target.checked })} className="accent-neon-blue" />
              <span className="text-xs text-slate-400">Pinned</span>
            </label>
            <Button onClick={() => saveEditMut.mutate()} disabled={saveEditMut.isPending}>Save Changes</Button>
          </div>
        )}

        {activeTab === 'replies' && (
          <div>
            <div className="space-y-3 mb-4 max-h-80 overflow-y-auto scrollbar-thin">
              {replies.map((reply: any) => (
                <div key={reply.id} className="bg-cyber-800 p-3 rounded-lg">
                  <div className="flex justify-between text-xs text-slate-400 mb-1">
                    <span className="font-semibold">{reply.author_username}</span>
                    <span>{new Date(reply.created_at).toLocaleString()}</span>
                  </div>
                  <div className="text-sm text-white" dangerouslySetInnerHTML={{ __html: marked.parse(reply.content) }} />
                </div>
              ))}
              {replies.length === 0 && <p className="text-slate-500 text-xs">No replies yet.</p>}
            </div>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-2 text-white text-sm"
              rows={2}
              placeholder="Write a reply..."
              value={replyContent}
              onChange={e => setReplyContent(e.target.value)}
            />
            <Button size="sm" className="mt-2" onClick={() => postReplyMut.mutate()} disabled={!replyContent.trim() || postReplyMut.isPending}>
              Post Reply
            </Button>
          </div>
        )}

        {activeTab === 'versions' && (
          <div className="space-y-2 max-h-80 overflow-y-auto scrollbar-thin">
            {versions.map((v: any) => (
              <div key={v.version} className="bg-cyber-800 p-3 rounded-lg flex justify-between items-center">
                <div>
                  <span className="text-white font-mono text-sm">Version {v.version}</span>
                  <p className="text-xs text-slate-400">Changed by: {v.changed_by || 'system'} · Reason: {v.change_reason || 'auto'}</p>
                </div>
                <Button size="sm" variant="ghost" onClick={() => alert(v.content_snapshot?.slice(0, 500))}>View</Button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}

// ---------- Main Page ----------
export default function ResearcherNotebookPage() {
  const { user, logout } = useAuthStore()
  const queryClient = useQueryClient()
  const [noteTitle, setNoteTitle] = useState('')
  const [noteContent, setNoteContent] = useState('')
  const [noteTags, setNoteTags] = useState('')
  const [noteRun, setNoteRun] = useState('')
  const [notePublic, setNotePublic] = useState(false)
  const [selectedNoteId, setSelectedNoteId] = useState<string | null>(null)
  const [filterTag, setFilterTag] = useState('')
  const [filterRun, setFilterRun] = useState('')
  const [filterPublic, setFilterPublic] = useState('all')
  const [pinnedOnly, setPinnedOnly] = useState(false)
  const [showWebhooks, setShowWebhooks] = useState(false)

  // Queries
  const { data: runs = [] } = useQuery({
    queryKey: ['runs'],
    queryFn: async () => (await api.get('/api/jobs/')).data, // or experiments
  })

  const { data: notesData, isLoading: notesLoading } = useQuery({
    queryKey: ['notes', filterTag, filterRun, filterPublic, pinnedOnly],
    queryFn: async () => {
      const params = new URLSearchParams()
      if (filterTag) params.set('tag', filterTag)
      if (filterRun) params.set('run', filterRun)
      if (filterPublic === 'public') params.set('public', 'true')
      else if (filterPublic === 'private') params.set('public', 'false')
      if (pinnedOnly) params.set('pinned', 'true')
      const url = `${ENDPOINTS.NOTES.LIST}?${params.toString()}`
      return (await api.get(url)).data
    },
  })

  const notes: Note[] = notesData?.results || []

  // Derived
  const uniqueTags = useMemo(() => {
    const all = new Set(notes.flatMap(n => n.tags || []))
    return all.size
  }, [notes])

  // Mutations
  const createNoteMut = useMutation({
    mutationFn: async () => {
      const tagsArray = noteTags.split(',').map(t => t.trim()).filter(Boolean)
      const payload = {
        title: noteTitle,
        content: noteContent,
        tags: tagsArray,
        run: noteRun || null,
        is_public: notePublic,
      }
      await api.post(ENDPOINTS.NOTES.LIST, payload)
    },
    onSuccess: () => {
      setNoteTitle('')
      setNoteContent('')
      setNoteTags('')
      setNoteRun('')
      setNotePublic(false)
      queryClient.invalidateQueries({ queryKey: ['notes'] })
    },
  })

  const deleteNoteMut = useMutation({
    mutationFn: async (id: string) => api.delete(ENDPOINTS.NOTES.DETAIL(id)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['notes'] }),
  })

  const reactMut = useMutation({
    mutationFn: async ({ id, reaction }: { id: string; reaction: string }) =>
      api.post(ENDPOINTS.NOTES.REACT(id), { reaction }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['notes'] }),
  })

  if (notesLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white">
            RESEARCHER <span className="text-neon-blue">NOTEBOOK</span>
          </h1>
          <p className="text-slate-400 text-sm">Capture insights, hypotheses, and observations</p>
        </div>
        <div className="flex gap-2">
          <Button variant="ghost" size="sm" onClick={() => setShowWebhooks(true)}>Webhooks</Button>
          <Button variant="ghost" size="sm" onClick={logout}>Logout</Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column – Create + Filters + Stats */}
        <div className="space-y-6">
          {/* Create Note */}
          <Card>
            <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <span className="w-1 h-4 bg-neon-blue rounded-full" />
              New Note
            </h2>
            <div className="space-y-3">
              <Input placeholder="Title (optional)" value={noteTitle} onChange={e => setNoteTitle(e.target.value)} />
              <textarea
                className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[120px] focus:border-neon-blue/50 resize-none"
                placeholder="Write in **Markdown**..."
                value={noteContent}
                onChange={e => setNoteContent(e.target.value)}
              />
              <Input placeholder="Tags (comma separated)" value={noteTags} onChange={e => setNoteTags(e.target.value)} />
              <Select
                label="Associated Run"
                options={[{ value: '', label: 'No specific run' }, ...runs.map((r: any) => ({ value: r.id, label: `${r.name || r.id.slice(0,8)} (${r.task_name || 'task'})` }))]}
                value={noteRun}
                onChange={e => setNoteRun(e.target.value)}
              />
              <label className="flex items-center gap-2 cursor-pointer">
                <input type="checkbox" checked={notePublic} onChange={e => setNotePublic(e.target.checked)} className="accent-neon-blue" />
                <span className="text-xs text-slate-400">Make public</span>
              </label>
              <Button onClick={() => createNoteMut.mutate()} disabled={!noteContent.trim() || createNoteMut.isPending} className="w-full">
                {createNoteMut.isPending ? 'Saving...' : 'Save Note'}
              </Button>
            </div>
          </Card>

          {/* Filters */}
          <Card>
            <h2 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
              <span className="w-1 h-4 bg-purple-500 rounded-full" />
              Filter Notes
            </h2>
            <div className="space-y-3">
              <Input placeholder="Filter by tag" value={filterTag} onChange={e => setFilterTag(e.target.value)} />
              <Select
                label="Run"
                options={[{ value: '', label: 'All runs' }, ...runs.map((r: any) => ({ value: r.id, label: r.name || r.id.slice(0,8) }))]}
                value={filterRun}
                onChange={e => setFilterRun(e.target.value)}
              />
              <Select
                label="Show"
                options={[
                  { value: 'all', label: 'All notes' },
                  { value: 'public', label: 'Public only' },
                  { value: 'private', label: 'My private' },
                ]}
                value={filterPublic}
                onChange={e => setFilterPublic(e.target.value)}
              />
              <div className="flex gap-2">
                <Button size="sm" onClick={() => queryClient.invalidateQueries({ queryKey: ['notes'] })}>Apply</Button>
                <Button size="sm" variant="ghost" onClick={() => { setFilterTag(''); setFilterRun(''); setFilterPublic('all'); }}>Clear</Button>
              </div>
            </div>
          </Card>

          {/* Quick Stats */}
          <Card>
            <h2 className="text-sm font-bold text-white mb-3">Notebook Stats</h2>
            <div className="grid grid-cols-2 gap-4 text-center">
              <div className="bg-cyber-800 rounded-lg p-3">
                <div className="text-2xl font-bold text-white">{notes.length}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Total Notes</div>
              </div>
              <div className="bg-cyber-800 rounded-lg p-3">
                <div className="text-2xl font-bold text-white">{uniqueTags}</div>
                <div className="text-[10px] text-slate-500 uppercase tracking-wider">Unique Tags</div>
              </div>
            </div>
          </Card>
        </div>

        {/* Right Column – Notes List */}
        <div className="lg:col-span-2">
          <Card>
            <div className="flex justify-between items-center mb-4">
              <h2 className="text-sm font-bold text-white flex items-center gap-2">
                <span className="w-1 h-4 bg-green-500 rounded-full" />
                All Notes
              </h2>
              <Button size="sm" variant={pinnedOnly ? 'secondary' : 'ghost'} onClick={() => setPinnedOnly(!pinnedOnly)}>
                📌 Pinned Only
              </Button>
            </div>
            {notes.length === 0 ? (
              <EmptyState message="No notes found. Create your first one!" />
            ) : (
              <div className="space-y-3 max-h-[70vh] overflow-y-auto scrollbar-thin">
                {notes.map(note => (
                  <NoteCard
                    key={note.id}
                    note={note}
                    onView={setSelectedNoteId}
                    onDelete={(id) => deleteNoteMut.mutate(id)}
                    onReact={(id, reaction) => reactMut.mutate({ id, reaction })}
                  />
                ))}
              </div>
            )}
          </Card>
        </div>
      </div>

      {/* Note Detail Modal */}
      <NoteModal noteId={selectedNoteId} onClose={() => setSelectedNoteId(null)} />

      {/* Webhook Management Modal */}
      {showWebhooks && <WebhookModal onClose={() => setShowWebhooks(false)} />}
    </div>
  )
}

// Webhook Modal Component
const WebhookModal = ({ onClose }: { onClose: () => void }) => {
  const queryClient = useQueryClient()
  const [url, setUrl] = useState('')
  const [events, setEvents] = useState('')

  const { data: webhooks = [] } = useQuery({
    queryKey: ['webhooks'],
    queryFn: async () => (await api.get('/api/notes/v1/webhooks/')).data,
  })

  const addMut = useMutation({
    mutationFn: async () => {
      const eventsArray = events.split(',').map(e => e.trim()).filter(Boolean)
      await api.post('/api/notes/v1/webhooks/', { url, events: eventsArray })
    },
    onSuccess: () => {
      setUrl('')
      setEvents('')
      queryClient.invalidateQueries({ queryKey: ['webhooks'] })
    },
  })

  const deleteMut = useMutation({
    mutationFn: async (id: string) => api.delete(`/api/notes/v1/webhooks/${id}/`),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['webhooks'] }),
  })

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 animate-fade-in">
      <div className="glass w-full max-w-lg p-6 space-y-4">
        <div className="flex justify-between items-center">
          <h2 className="text-xl font-black text-white">Webhook Subscriptions</h2>
          <Button variant="ghost" size="sm" onClick={onClose}>✕</Button>
        </div>
        <div className="space-y-2 max-h-48 overflow-y-auto scrollbar-thin">
          {webhooks.map((wh: any) => (
            <div key={wh.id} className="flex justify-between items-center bg-cyber-800 p-3 rounded-lg">
              <div className="text-xs">
                <code className="text-neon-blue">{wh.url}</code>
                <p className="text-slate-400">{wh.events.join(', ')}</p>
              </div>
              <Button size="sm" variant="danger" onClick={() => deleteMut.mutate(wh.id)}>Delete</Button>
            </div>
          ))}
        </div>
        <div className="space-y-2">
          <Input placeholder="Webhook URL" value={url} onChange={e => setUrl(e.target.value)} />
          <Input placeholder="Events (comma: note.created,note.updated)" value={events} onChange={e => setEvents(e.target.value)} />
          <Button onClick={() => addMut.mutate()} disabled={!url || addMut.isPending}>Add Webhook</Button>
        </div>
      </div>
    </div>
  )
}