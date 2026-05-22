import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Play, Square, Trash2, RefreshCw } from 'lucide-react'

export function EndpointCard({ endpoint, onRefresh }: any) {
  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-start"><div><h3 className="font-bold text-white">{endpoint.name}</h3><div className="text-xs text-slate-400">{endpoint.model_id} v{endpoint.model_version}</div><Badge variant={endpoint.status === 'running' ? 'success' : endpoint.status === 'failed' ? 'danger' : 'warning'} className="mt-1">{endpoint.status}</Badge></div><div className="flex gap-1"><Button size="sm" variant="ghost" onClick={() => window.open(endpoint.url, '_blank')}>Test</Button><Button size="sm" variant="ghost" onClick={onRefresh}><RefreshCw size={12} /></Button><Button size="sm" variant="ghost"><Trash2 size={12} className="text-red-400" /></Button></div></div>
      <div className="grid grid-cols-2 gap-2 mt-3 text-sm"><div>Requests/s: {endpoint.metrics?.requests_per_second || 0}</div><div>Latency: {endpoint.metrics?.avg_latency_ms || 0}ms</div><div>Error rate: {(endpoint.metrics?.error_rate || 0) * 100}%</div><div>Replicas: {endpoint.config.replicas}</div></div>
    </TerminalCard>
  )
}