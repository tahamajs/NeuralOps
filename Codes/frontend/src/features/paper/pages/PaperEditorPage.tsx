import { useParams } from 'react-router-dom'
import { useState, useEffect } from 'react'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function PaperEditorPage() {
  const { runId } = useParams<{ runId: string }>()
  const [source, setSource] = useState('')
  const [saving, setSaving] = useState(false)
  const [message, setMessage] = useState('')

  useEffect(() => {
    const fetchSource = async () => {
      try {
        const res = await api.get(`/api/runs/${runId}/paper/`)
        // If it returns text, assume it's LaTeX
        if (res.headers['content-type']?.includes('text')) {
          const text = typeof res.data === 'string' ? res.data : JSON.stringify(res.data)
          setSource(text)
        }
      } catch {}
    }
    fetchSource()
  }, [runId])

  const handleSave = async () => {
    setSaving(true)
    try {
      await api.post(`/api/runs/${runId}/paper_source/`, { source })
      setMessage('Saved successfully')
    } catch (err) {
      setMessage('Save failed')
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="space-y-6 animate-fade-in max-w-5xl mx-auto">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        PAPER <span className="text-neon-blue">EDITOR</span>
      </h1>
      <Card>
        <textarea
          className="w-full h-[500px] bg-cyber-800 text-slate-300 font-mono text-sm p-4 rounded-lg border border-white/10 focus:border-neon-blue/50"
          value={source}
          onChange={(e) => setSource(e.target.value)}
          placeholder="LaTeX source..."
        />
        <div className="flex justify-between items-center mt-4">
          <span className="text-xs text-slate-400">{message}</span>
          <Button onClick={handleSave} disabled={saving}>{saving ? 'Saving...' : 'Save Source'}</Button>
        </div>
      </Card>
    </div>
  )
}