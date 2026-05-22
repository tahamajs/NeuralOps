// src/features/coscientist/components/AgentActivityPanel.tsx
import { Agent } from '../types'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Loader2, CheckCircle, AlertCircle } from 'lucide-react'

interface AgentActivityPanelProps { agents: Agent[] }

export function AgentActivityPanel({ agents }: AgentActivityPanelProps) {
  const getIcon = (status: Agent['status']) => {
    switch (status) {
      case 'working': return <Loader2 size={14} className="animate-spin text-yellow-400" />
      case 'completed': return <CheckCircle size={14} className="text-green-400" />
      case 'error': return <AlertCircle size={14} className="text-red-400" />
      default: return <div className="w-3.5 h-3.5 rounded-full bg-slate-600" />
    }
  }
  const getStatusColor = (status: Agent['status']) => {
    switch (status) { case 'working': return 'text-yellow-400'; case 'completed': return 'text-green-400'; case 'error': return 'text-red-400'; default: return 'text-slate-500' }
  }
  return (
    <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-neon-mint animate-pulse" />Agent Orchestra</h3>
      <div className="space-y-3">{agents.map(agent => (<div key={agent.id} className="flex justify-between items-center"><div className="flex items-center gap-2">{getIcon(agent.status)}<span className="text-sm text-white">{agent.name}</span></div><div className="text-xs text-slate-500">{agent.status}</div></div>))}</div>
    </TerminalCard>
  )
}