// src/features/integrations/pages/SandboxSessionView.tsx
import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { integrationsAPI } from '../services/api'
import { TerminalEmulator } from '../components/TerminalEmulator'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Terminal, Plus, Trash2, Play } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function SandboxSessionView({ onClose }: { onClose?: () => void }) {
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [environment, setEnvironment] = useState('python')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: sessions, isLoading } = useQuery({
    queryKey: ['sandbox-sessions'],
    queryFn: () => integrationsAPI.getSandboxSessions(),
    refetchInterval: 5000,
  })

  const createSession = useMutation({
    mutationFn: () => integrationsAPI.createSandboxSession(environment),
    onSuccess: (newSession) => {
      queryClient.invalidateQueries({ queryKey: ['sandbox-sessions'] })
      setActiveSessionId(newSession.id)
      success('Sandbox created')
    },
  })

  const terminateSession = useMutation({
    mutationFn: (id: string) => integrationsAPI.terminateSandboxSession(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['sandbox-sessions'] }),
  })

  if (isLoading) return <Spinner />

  const activeSession = sessions?.find(s => s.id === activeSessionId)

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Terminal size={20} className="text-neon-mint" /> Secure Sandbox</h1><p className="text-xs text-slate-400">Isolated execution environments</p></div>
        <div className="flex gap-2">{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-hidden p-4"><div className="grid lg:grid-cols-3 gap-4 h-full"><div className="space-y-3 overflow-y-auto"><div className="flex gap-2"><Select options={[{ value: 'python', label: 'Python 3.11' }, { value: 'node', label: 'Node.js 20' }, { value: 'bash', label: 'Bash' }]} value={environment} onChange={e => setEnvironment(e.target.value)} className="flex-1" /><Button size="sm" onClick={() => createSession.mutate()}><Plus size={14} className="mr-1" /> New</Button></div>
          {sessions?.map(s => (<div key={s.id} className={`p-3 rounded-lg cursor-pointer border ${activeSessionId === s.id ? 'border-neon-blue bg-neon-blue/10' : 'border-white/10 hover:bg-white/5'}`} onClick={() => setActiveSessionId(s.id)}><div className="flex justify-between items-center"><span className="font-mono text-sm text-white">{s.name.slice(0, 8)}</span><Badge variant={s.status === 'active' ? 'success' : 'secondary'}>{s.status}</Badge></div><div className="text-xs text-slate-400 mt-1">{s.environment} · {new Date(s.createdAt).toLocaleString()}</div><div className="text-right mt-2"><button onClick={(e) => { e.stopPropagation(); terminateSession.mutate(s.id) }} className="text-red-400 text-xs">Terminate</button></div></div>))}
        </div>
        <div className="lg:col-span-2">{activeSession ? <TerminalEmulator sessionId={activeSession.id} /> : <TerminalCard className="h-full flex items-center justify-center text-slate-500">Select or create a sandbox session</TerminalCard>}</div>
      </div></div>
    </div>
  )
}