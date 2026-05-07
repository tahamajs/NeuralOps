import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { Link, useSearchParams } from 'react-router-dom'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'
import { Badge } from '../../../components/ui/Badge'
import { ResearcherNote } from '../types'

const VIEWS = ['all', 'mine', 'shared', 'pinned', 'public', 'archived'] as const
type ViewType = typeof VIEWS[number]

export default function NotesListPage() {
  const [searchParams, setSearchParams] = useSearchParams()
  const [view, setView] = useState<ViewType>((searchParams.get('view') as ViewType) || 'all')
  const [search, setSearch] = useState(searchParams.get('q') || '')
  const [debouncedSearch, setDebouncedSearch] = useState(search)
  const [selectedTags, setSelectedTags] = useState<string[]>([])
  const [sortBy, setSortBy] = useState<'updated' | 'created' | 'title'>('updated')
  const queryClient = useQueryClient()

  // Debounce search
  const handleSearch = (val: string) => {
    setSearch(val)
    const timer = setTimeout(() => {
      setDebouncedSearch(val)
      setSearchParams(val ? { view, q: val } : { view })
    }, 300)
    return () => clearTimeout(timer)
  }

  const { data, isLoading } = useQuery({
    queryKey: ['notes', view],
    queryFn: async () => {
      const endpoints: Record<ViewType, string> = {
        all: ENDPOINTS.NOTES.LIST,
        mine: `${ENDPOINTS.NOTES.LIST}?author_username=me`,
        shared: `${ENDPOINTS.NOTES.LIST}?shared=true`,
        pinned: `${ENDPOINTS.NOTES.LIST}?pinned=true`,
        public: `${ENDPOINTS.NOTES.LIST}?is_public=true`,
        archived: `${ENDPOINTS.NOTES.LIST}?archived=true`,
      }
      return (await api.get(endpoints[view])).data
    },
    refetchInterval: 10000,
  })

  const notes: ResearcherNote[] = data?.results || []

  // Collect all unique tags
  const allTags = Array.from(new Set(notes.flatMap((n: ResearcherNote) => n.tags || [])))

  // Filter & Sort
  const filtered = notes
    .filter((n: ResearcherNote) => {
      if (debouncedSearch) {
        const q = debouncedSearch.toLowerCase()
        return (
          (n.title || '').toLowerCase().includes(q) ||
          (n.content || '').toLowerCase().includes(q) ||
          (n.tags || []).some((t: string) => t.toLowerCase().includes(q))
        )
      }
      if (selectedTags.length > 0) {
        return selectedTags.some((t) => (n.tags || []).includes(t))
      }
      return true
    })
    .sort((a: ResearcherNote, b: ResearcherNote) => {
      if (a.pinned && !b.pinned) return -1
      if (!a.pinned && b.pinned) return 1
      if (sortBy === 'title') return (a.title || '').localeCompare(b.title || '')
      const aDate = sortBy === 'created' ? a.created_at : a.updated_at
      const bDate = sortBy === 'created' ? b.created_at : b.updated_at
      return new Date(bDate).getTime() - new Date(aDate).getTime()
    })

  // Delete mutation
  const deleteMut = useMutation({
    mutationFn: (id: string) => api.delete(ENDPOINTS.NOTES.DETAIL(id)),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['notes'] }),
  })

  // Pin mutation
  const pinMut = useMutation({
    mutationFn: async ({ id, pinned }: { id: string; pinned: boolean }) =>
      api.patch(ENDPOINTS.NOTES.DETAIL(id), { pinned: !pinned }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['notes'] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      {/* ── Header ── */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <h1 className="text-3xl font-black text-white tracking-tighter">
          RESEARCH <span className="text-neon-blue">NOTES</span>
        </h1>
        <Link to="/notes/new">
          <Button size="md">+ New Note</Button>
        </Link>
      </div>

      {/* ── View Tabs ── */}
      <div className="flex gap-1 p-1 bg-white/5 rounded-xl w-fit overflow-x-auto scrollbar-thin">
        {VIEWS.map((v) => (
          <button
            key={v}
            onClick={() => { setView(v); setSearchParams({ view: v }) }}
            className={`px-4 py-2 rounded-lg text-xs font-black uppercase tracking-widest transition-all whitespace-nowrap ${
              view === v ? 'bg-neon-blue text-white shadow' : 'text-slate-400 hover:text-white'
            }`}
          >
            {v}
          </button>
        ))}
      </div>

      {/* ── Search, Sort, Tags ── */}
      <div className="flex flex-wrap gap-3 items-center">
        <Input
          placeholder="Search notes..."
          value={search}
          onChange={(e) => handleSearch(e.target.value)}
          className="flex-1 min-w-[200px]"
        />
        <select
          className="bg-white/5 border border-white/10 rounded-xl px-3 py-2 text-xs text-white"
          value={sortBy}
          onChange={(e) => setSortBy(e.target.value as any)}
        >
          <option value="updated">Recently Updated</option>
          <option value="created">Recently Created</option>
          <option value="title">Title A-Z</option>
        </select>
        <Link to="/notes/statistics">
          <Button variant="ghost" size="sm">Stats</Button>
        </Link>
      </div>

      {/* ── Tag Cloud ── */}
      {allTags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {allTags.map((tag) => (
            <button
              key={tag}
              onClick={() => setSelectedTags((prev) =>
                prev.includes(tag) ? prev.filter((t) => t !== tag) : [...prev, tag]
              )}
              className={`px-2 py-1 rounded-full text-[10px] font-bold uppercase tracking-wide border transition-all ${
                selectedTags.includes(tag)
                  ? 'bg-neon-blue/20 border-neon-blue/50 text-neon-blue'
                  : 'bg-white/5 border-white/10 text-slate-400 hover:bg-white/10'
              }`}
            >
              {tag}
              {selectedTags.includes(tag) && ' ✕'}
            </button>
          ))}
          {selectedTags.length > 0 && (
            <button onClick={() => setSelectedTags([])} className="text-[10px] text-red-400">Clear all</button>
          )}
        </div>
      )}

      {/* ── Notes Grid ── */}
      {filtered.length === 0 ? (
        <EmptyState message={search ? `No notes matching "${search}"` : "No notes yet. Create one!"} />
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {filtered.map((note) => (
            <Link key={note.id} to={`/notes/${note.id}`} className="block group">
              <Card hover className="h-full flex flex-col relative">
                {/* Pin indicator */}
                {note.pinned && (
                  <div className="absolute top-2 right-2 text-yellow-400 text-xs">📌</div>
                )}
                {/* Shared indicator */}
                {note.shared_with?.length > 0 && (
                  <div className="absolute top-2 right-2 text-neon-blue text-xs" title={`Shared with ${note.shared_with.join(', ')}`}>
                    👥
                  </div>
                )}
                <h3 className="text-white font-semibold text-sm group-hover:text-neon-blue transition-colors line-clamp-1">
                  {note.title || 'Untitled'}
                </h3>
                <p className="text-xs text-slate-400 mt-1.5 line-clamp-3 flex-1">
                  {note.content?.slice(0, 180)}
                </p>
                <div className="flex flex-wrap gap-1 mt-2">
                  {note.tags?.slice(0, 4).map((tag) => (
                    <span key={tag} className="px-1.5 py-0.5 rounded bg-white/5 text-[9px] text-slate-500 uppercase tracking-wide">
                      {tag}
                    </span>
                  ))}
                </div>
                <div className="flex justify-between items-center mt-3 pt-2 border-t border-white/5 text-[10px] text-slate-500">
                  <span>{note.author_username}</span>
                  <span>{new Date(note.updated_at).toLocaleDateString()}</span>
                </div>
                {/* Quick actions */}
                <div className="absolute bottom-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
                  <button
                    onClick={(e) => { e.preventDefault(); pinMut.mutate({ id: note.id, pinned: note.pinned }) }}
                    className="p-1 rounded hover:bg-white/10 text-xs"
                    title={note.pinned ? 'Unpin' : 'Pin'}
                  >
                    {note.pinned ? '📌' : '📍'}
                  </button>
                  <button
                    onClick={(e) => { e.preventDefault(); deleteMut.mutate(note.id) }}
                    className="p-1 rounded hover:bg-red-500/20 text-xs"
                    title="Delete"
                  >
                    🗑️
                  </button>
                </div>
              </Card>
            </Link>
          ))}
        </div>
      )}
    </div>
  )
}