import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import { Bot, Play, Pause } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

export default function AgentDashboardPage() {
  const { sessionId } = useParams<{ sessionId: string }>()

  const { data: session, isLoading } = useQuery({
    queryKey: ['agentSession', sessionId],
    queryFn: async () => (await api.get(`/api/agents/sessions/${sessionId}/`)).data,
    refetchInterval: 3000,
    enabled: !!sessionId,
  })

  if (isLoading || !session) return <Spinner />

  const handlePauseResume = async () => {
    const newStatus = session.status === 'active' ? 'paused' : 'active'
    await api.patch(`/api/agents/sessions/${sessionId}/`, { status: newStatus })
  }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
            <Bot size={24} className="text-neon-blue" /> {session.name || 'Unnamed Agent'}
          </h1>
          <p className="text-slate-400 text-sm mt-1">{session.objective}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant={session.status === 'active' ? 'secondary' : 'primary'}
            onClick={handlePauseResume}
          >
            {session.status === 'active' ? <Pause size={14} className="mr-1" /> : <Play size={14} className="mr-1" />}
            {session.status === 'active' ? 'Pause' : 'Resume'}
          </Button>
          <Badge status={session.status === 'active' ? 'running' : session.status === 'completed' ? 'completed' : 'queued'}>
            {session.status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card className="text-center p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Cycles</div>
          <div className="text-2xl font-bold text-white mt-1">{session.cycles_completed}/{session.max_cycles}</div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Experiments</div>
          <div className="text-2xl font-bold text-white mt-1">{session.experiments_run || 0}</div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Best Metric</div>
          <div className="text-2xl font-bold text-green-400 mt-1">{session.best_metric_value?.toFixed(4) || '—'}</div>
        </Card>
        <Card className="text-center p-4">
          <div className="text-xs text-slate-400 uppercase tracking-wider">Cost</div>
          <div className="text-2xl font-bold text-white mt-1">${((session.total_llm_cost_cents || 0) / 100).toFixed(2)}</div>
        </Card>
      </div>

      <Card>
        <h3 className="text-sm font-display font-bold text-white mb-4">Agent Memory</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Working Memory</h4>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-white whitespace-pre-wrap max-h-48 overflow-y-auto scrollbar-thin">
              {session.memory?.working_memory || 'Empty'}
            </pre>
          </div>
          <div>
            <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">Long‑Term Summary</h4>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-white whitespace-pre-wrap max-h-48 overflow-y-auto scrollbar-thin">
              {session.memory?.long_term_summary || 'Empty'}
            </pre>
          </div>
        </div>
      </Card>

      <Card>
        <h3 className="text-sm font-display font-bold text-white mb-4">Actions Log</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-400 uppercase tracking-wider">
                <th className="py-2 px-2">Cycle</th>
                <th className="py-2 px-2">Type</th>
                <th className="py-2 px-2">Tool</th>
                <th className="py-2 px-2">Summary</th>
                <th className="py-2 px-2">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {session.actions?.map((action: any) => (
                <tr key={action.id}>
                  <td className="py-2 px-2 text-white">{action.cycle}</td>
                  <td className="py-2 px-2"><Badge>{action.action_type}</Badge></td>
                  <td className="py-2 px-2 text-neon-blue font-mono">{action.tool_name || '—'}</td>
                  <td className="py-2 px-2 text-slate-400 max-w-xs truncate">{action.output_summary?.slice(0, 80)}</td>
                  <td className="py-2 px-2 text-slate-500">{formatDistanceToNow(new Date(action.created_at), { addSuffix: true })}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}