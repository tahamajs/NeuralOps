import { useParams } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { useState, useEffect } from 'react'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'

export default function UpdateRunConfigPage() {
  const { runId } = useParams<{ runId: string }>()
  const [configText, setConfigText] = useState('')
  const { data: run, isLoading } = useQuery({
    queryKey: ['run', runId],
    queryFn: async () => (await api.get(`/api/experiments/${runId}/`)).data,
  })

  useEffect(() => {
    if (run) setConfigText(JSON.stringify(run.config, null, 2))
  }, [run])

  const updateMut = useMutation({
    mutationFn: async () => {
      const config = JSON.parse(configText)
      return api.patch(`/api/runs/${runId}/config/`, { config })
    },
    onSuccess: () => alert('Config updated'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-3xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        UPDATE <span className="text-neon-blue">CONFIG</span>
      </h1>
      <Card>
        <p className="text-sm text-slate-400 mb-4">Edit the JSON configuration for this run.</p>
        <textarea
          className="w-full bg-cyber-800 text-white font-mono text-xs p-4 rounded-lg h-96 border border-white/10 focus:border-neon-blue/50"
          value={configText}
          onChange={(e) => setConfigText(e.target.value)}
        />
        <Button className="mt-4" onClick={() => updateMut.mutate()} disabled={updateMut.isPending}>
          Save Changes
        </Button>
      </Card>
    </div>
  )
}