import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { ResearcherNote } from '../types'

export default function NoteCreatePage() {
  const navigate = useNavigate()
  const [form, setForm] = useState({ title: '', content: '', tags: '', is_public: false })

  const createMutation = useMutation({
    mutationFn: async () => {
      const tagsArray = form.tags.split(',').map(t => t.trim()).filter(Boolean)
      const res = await api.post(ENDPOINTS.NOTES.LIST, { ...form, tags: tagsArray })
      return res.data
    },
    onSuccess: (data: ResearcherNote) => {
      navigate(`/notes/${data.id}`)
    },
  })

  return (
    <div className="max-w-3xl mx-auto animate-fade-in space-y-6">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        NEW <span className="text-neon-blue">NOTE</span>
      </h1>
      <Card>
        <div className="space-y-4">
          <Input
            label="Title"
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
          />
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">Content</label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white min-h-[300px] focus:outline-none focus:border-neon-blue/50"
              value={form.content}
              onChange={(e) => setForm({ ...form, content: e.target.value })}
            />
          </div>
          <Input
            label="Tags (comma separated)"
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
          />
          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="public"
              checked={form.is_public}
              onChange={(e) => setForm({ ...form, is_public: e.target.checked })}
            />
            <label htmlFor="public" className="text-xs text-slate-400">Public note</label>
          </div>
          <Button onClick={() => createMutation.mutate()} disabled={createMutation.isPending}>
            Create Note
          </Button>
        </div>
      </Card>
    </div>
  )
}