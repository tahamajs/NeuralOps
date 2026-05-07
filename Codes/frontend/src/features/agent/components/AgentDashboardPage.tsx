import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { formatDistanceToNow } from 'date-fns'

export default function AgentDashboardPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const { data: session, isLoading } = useQuery({
    queryKey: ['agentSession', sessionId],
    queryFn: async () => (await api.get(`/api/agents/sessions/${sessionId}/`)).data,
    refetchInterval: 3000,
  })

  if (isLoading || !session) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-display font-bold text-white">🤖 {session.name}</h1>
          <p className="text-slate-400 text-sm">{session.objective}</p>
        </div>
        <div>
          {session.status === 'active' && (
            <Button size="sm" variant="secondary" onClick={() => api.patch(`/api/agents/sessions/${sessionId}/`, {status: 'paused'})}>⏸ Pause</Button>
          )}
          {session.status === 'paused' && (
            <Button size="sm" onClick={() => api.patch(`/api/agents/sessions/${sessionId}/`, {status: 'active'})}>▶ Resume</Button>
          )}
          <Badge status={session.status === 'active' ? 'running' : session.status === 'completed' ? 'completed' : 'queued'} className="ml-2">
            {session.status}
          </Badge>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-4">
        <Card className="text-center"><div className="text-xs text-slate-400">Cycles</div><div className="text-2xl font-bold text-white">{session.cycles_completed}/{session.max_cycles}</div></Card>
        <Card className="text-center"><div className="text-xs text-slate-400">Experiments</div><div className="text-2xl font-bold text-white">{session.experiments_run}</div></Card>
        <Card className="text-center"><div className="text-xs text-slate-400">Best Metric</div><div className="text-2xl font-bold text-green-400">{session.best_metric_value?.toFixed(4) || '—'}</div></Card>
        <Card className="text-center"><div className="text-xs text-slate-400">Cost</div><div className="text-2xl font-bold text-white">${(session.total_llm_cost_cents / 100).toFixed(2)}</div></Card>
      </div>

      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Agent Memory</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-1">Working Memory</h4>
            <pre className="bg-cyber-800 p-3 rounded text-xs text-white whitespace-pre-wrap">{session.memory?.working_memory || 'Empty'}</pre>
          </div>
          <div>
            <h4 className="text-xs text-slate-400 uppercase tracking-wider mb-1">Long-Term Summary</h4>
            <pre className="bg-cyber-800 p-3 rounded text-xs text-white whitespace-pre-wrap">{session.memory?.long_term_summary || 'Empty'}</pre>
          </div>
        </div>
      </Card>

      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Actions Log</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="py-2 px-2">Cycle</th><th className="py-2 px-2">Type</th><th className="py-2 px-2">Tool</th><th className="py-2 px-2">Summary</th><th className="py-2 px-2">Time</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {session.actions?.map((action: any) => (
                <tr key={action.id}>
                  <td className="py-2 px-2">{action.cycle}</td>
                  <td className="py-2 px-2"><Badge>{action.action_type}</Badge></td>
                  <td className="py-2 px-2 text-neon-blue">{action.tool_name || '—'}</td>
                  <td className="py-2 px-2 text-[10px] max-w-xs truncate">{action.output_summary?.slice(0,80)}</td>
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