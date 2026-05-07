import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'

export default function NoteCreatePage() {
  const navigate = useNavigate()
  const [form, setForm] = useState({
    title: '',
    content: '',
    tags: '',
    is_public: false,
    pinned: false,
  })

  const createMut = useMutation({
    mutationFn: async () => {
      const tagsArray = form.tags.split(',').map((t) => t.trim()).filter(Boolean)
      const res = await api.post(ENDPOINTS.NOTES.LIST, {
        ...form,
        tags: tagsArray,
      })
      return res.data
    },
    onSuccess: (data) => navigate(`/notes/${data.id}`),
    onError: (err: any) => alert(err.response?.data?.error || 'Failed to create note'),
  })

  return (
    <div className="max-w-3xl mx-auto space-y-6 animate-fade-in">
      <h1 className="text-3xl font-black text-white tracking-tighter">
        NEW <span className="text-neon-blue">NOTE</span>
      </h1>
      <Card>
        <div className="space-y-5">
          <Input
            label="Title"
            value={form.title}
            onChange={(e) => setForm({ ...form, title: e.target.value })}
            placeholder="Give your note a title..."
            className="text-lg font-bold"
          />
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">
              Content <span className="text-neon-blue">(Markdown supported)</span>
            </label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-4 text-white text-sm min-h-[400px] focus:border-neon-blue/50 transition-colors scrollbar-thin font-mono"
              value={form.content}
              onChange={(e) => setForm({ ...form, content: e.target.value })}
              placeholder={`# Research Notes\n\nStart writing your findings here...\n\n## Key Observations\n- ...\n\n## References\n- ...`}
            />
          </div>
          <Input
            label="Tags (comma separated)"
            value={form.tags}
            onChange={(e) => setForm({ ...form, tags: e.target.value })}
            placeholder="ml, attention, transformer"
          />
          <div className="flex items-center gap-6">
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.is_public}
                onChange={(e) => setForm({ ...form, is_public: e.target.checked })}
                className="accent-neon-blue"
              />
              <span className="text-xs text-slate-400">Public (visible to all users)</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={form.pinned}
                onChange={(e) => setForm({ ...form, pinned: e.target.checked })}
                className="accent-neon-blue"
              />
              <span className="text-xs text-slate-400">📌 Pin to top</span>
            </label>
          </div>
          <Button
            size="lg"
            className="w-full mt-4"
            onClick={() => createMut.mutate()}
            disabled={createMut.isPending}
          >
            {createMut.isPending ? 'Creating...' : '✨ Create Note'}
          </Button>
        </div>
      </Card>
    </div>
  )
}