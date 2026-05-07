import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'

export default function FloatingNotesEditor() {
  const queryClient = useQueryClient()
  const [selectedNoteId, setSelectedNoteId] = useState<string>('')
  const [content, setContent] = useState('')
  const [title, setTitle] = useState('')

  const { data: notesList } = useQuery({
    queryKey: ['notes'],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.LIST)).data,
  })

  const { data: note } = useQuery({
    queryKey: ['note', selectedNoteId],
    queryFn: async () => (await api.get(ENDPOINTS.NOTES.DETAIL(selectedNoteId))).data,
    enabled: !!selectedNoteId,
    onSuccess: (data: any) => {
      setTitle(data.title || '')
      setContent(data.content || '')
    },
  })

  const saveMut = useMutation({
    mutationFn: async () => {
      if (selectedNoteId) {
        return api.patch(ENDPOINTS.NOTES.DETAIL(selectedNoteId), { title, content })
      } else {
        return api.post(ENDPOINTS.NOTES.LIST, { title, content })
      }
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['notes'] })
    },
  })

  const noteOptions = (notesList?.results || []).map((n: any) => ({
    value: n.id,
    label: n.title || `Note ${n.id.slice(0, 8)}`,
  }))

  return (
    <div className="p-4 space-y-4 bg-cyber-900 h-full overflow-y-auto">
      <Select
        label="Quick Note Select"
        options={[{ value: '', label: 'New Note...' }, ...noteOptions]}
        value={selectedNoteId}
        onChange={(e) => setSelectedNoteId(e.target.value)}
      />
      <Input
        placeholder="Note title..."
        value={title}
        onChange={(e) => setTitle(e.target.value)}
      />
      <textarea
        className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white text-sm min-h-[250px] focus:border-neon-blue/50 transition-colors resize-none"
        value={content}
        onChange={(e) => setContent(e.target.value)}
        placeholder="Write your notes here..."
      />
      <Button size="sm" onClick={() => saveMut.mutate()} disabled={saveMut.isPending}>
        {saveMut.isPending ? 'Saving...' : '💾 Save Note'}
      </Button>
    </div>
  )
}