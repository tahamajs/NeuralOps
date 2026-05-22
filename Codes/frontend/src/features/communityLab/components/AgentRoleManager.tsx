// src/features/communityLab/components/AgentRoleManager.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { communityLabAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Bot, Plus, Play, Square, MoreVertical, Settings, Trash2 } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

const AGENT_TYPES = ['Researcher', 'Coder', 'Critic', 'Data Scientist', 'Reviewer', 'Orchestrator']

export function AgentRoleManager({ labId }: { labId: string }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [selectedAgent, setSelectedAgent] = useState<any>(null)
  const [newAgentName, setNewAgentName] = useState('')
  const [newAgentType, setNewAgentType] = useState('Researcher')
  const [newAgentPrompt, setNewAgentPrompt] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: agents, isLoading } = useQuery({
    queryKey: ['lab-agents', labId],
    queryFn: () => communityLabAPI.getLabAgents(labId),
  })

  const createAgent = useMutation({
    mutationFn: () => communityLabAPI.createAgent(labId, { name: newAgentName, type: newAgentType, system_prompt: newAgentPrompt }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['lab-agents', labId] })
      setShowCreateModal(false)
      setNewAgentName('')
      setNewAgentPrompt('')
      success('Agent created', `${newAgentName} is ready to work.`)
    },
    onError: () => error('Failed to create agent'),
  })

  const startAgent = useMutation({
    mutationFn: (agentId: string) => communityLabAPI.startAgent(labId, agentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['lab-agents', labId] }),
  })

  const stopAgent = useMutation({
    mutationFn: (agentId: string) => communityLabAPI.stopAgent(labId, agentId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['lab-agents', labId] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-4">
      <div className="flex justify-between items-center"><h3 className="text-sm font-bold text-white">Active Agents</h3><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> Add Agent</Button></div>
      <div className="grid md:grid-cols-2 gap-3">
        {agents?.map(agent => (
          <TerminalCard key={agent.id} className="p-3">
            <div className="flex justify-between items-start"><div><div className="flex items-center gap-2"><Bot size={16} className="text-neon-mint" /><span className="font-bold text-white">{agent.name}</span><Badge variant="secondary" className="text-[10px]">{agent.type}</Badge></div><div className="text-xs text-slate-400 mt-1 line-clamp-2">{agent.system_prompt?.slice(0, 100)}</div></div><div className="flex gap-1">
              {agent.status === 'running' ? <button onClick={() => stopAgent.mutate(agent.id)}><Square size={14} className="text-red-400" /></button> : <button onClick={() => startAgent.mutate(agent.id)}><Play size={14} className="text-green-400" /></button>}
              <button onClick={() => setSelectedAgent(agent)}><Settings size={14} className="text-slate-400" /></button>
            </div></div>
            <div className="flex gap-2 mt-2 text-xs text-slate-500"><span>Tasks: {agent.tasks_completed}</span><span>Success: {agent.success_rate}%</span></div>
          </TerminalCard>
        ))}
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create New Agent"><div className="space-y-4"><Input label="Agent Name" value={newAgentName} onChange={e => setNewAgentName(e.target.value)} /><select value={newAgentType} onChange={e => setNewAgentType(e.target.value)} className="w-full bg-white/5 border border-white/10 rounded p-2">{AGENT_TYPES.map(t => <option key={t}>{t}</option>)}</select><Input label="System Prompt" value={newAgentPrompt} onChange={e => setNewAgentPrompt(e.target.value)} placeholder="You are a helpful research assistant..." /><Button onClick={() => createAgent.mutate()} disabled={!newAgentName}>Create Agent</Button></div></Modal>
    </div>
  )
}