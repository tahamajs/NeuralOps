import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { anomalyAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { Tabs, TabList, Tab, TabPanel } from '../../../components/ui/Tabs'
import { SuggestedFixes } from '../components/SuggestedFixes'
import { MetricChart } from '../components/MetricChart'
import { CheckCircle, AlertTriangle, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function AnomalyDetailPage({ onClose }: { onClose?: () => void }) {
  const { anomalyId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: anomaly, isLoading } = useQuery({
    queryKey: ['anomaly', anomalyId],
    queryFn: () => anomalyAPI.getAnomaly(anomalyId!),
    enabled: !!anomalyId,
  })

  const resolveMutation = useMutation({
    mutationFn: () => anomalyAPI.resolveAnomaly(anomalyId!, true),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['anomaly', anomalyId] })
      success('Anomaly resolved')
    },
  })

  const reanalyzeMutation = useMutation({
    mutationFn: () => anomalyAPI.detectAnomaly(anomaly!.run_id),
    onSuccess: () => success('Re‑analysis triggered'),
  })

  if (isLoading) return <Spinner />
  if (!anomaly) return <div className="p-4 text-red-400">Anomaly not found</div>

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white">Anomaly Details</h1><p className="text-sm text-slate-400">{anomaly.run_name || anomaly.run_id}</p></div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => reanalyzeMutation.mutate()}><RefreshCw size={14} /> Re‑analyze</Button>
          {anomaly.status !== 'resolved' && <Button size="sm" variant="success" onClick={() => resolveMutation.mutate()}><CheckCircle size={14} /> Resolve</Button>}
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid grid-cols-2 gap-4">
          <TerminalCard className="p-4"><div className="text-slate-400 text-xs">Severity</div><Badge variant={anomaly.severity === 'critical' ? 'danger' : anomaly.severity === 'high' ? 'danger' : anomaly.severity === 'medium' ? 'warning' : 'info'}>{anomaly.severity}</Badge></TerminalCard>
          <TerminalCard className="p-4"><div className="text-slate-400 text-xs">Status</div><Badge variant={anomaly.status === 'resolved' ? 'success' : 'warning'}>{anomaly.status}</Badge></TerminalCard>
          <TerminalCard className="p-4"><div className="text-slate-400 text-xs">Detected</div><div>{new Date(anomaly.detected_at).toLocaleString()}</div></TerminalCard>
          {anomaly.resolved_at && <TerminalCard className="p-4"><div className="text-slate-400 text-xs">Resolved</div><div>{new Date(anomaly.resolved_at).toLocaleString()} by {anomaly.resolved_by}</div></TerminalCard>}
        </div>
        <TerminalCard className="p-4"><div className="text-sm font-bold text-white mb-1">Summary</div><div className="text-slate-300">{anomaly.summary}</div></TerminalCard>
        <Tabs defaultValue="details">
          <TabList><Tab value="details">Details</Tab><Tab value="fixes">Suggested Fixes</Tab><Tab value="chart">Metric Chart</Tab></TabList>
          <TabPanel value="details">
            <TerminalCard className="p-4">
              <pre className="text-xs text-slate-300 whitespace-pre-wrap overflow-auto max-h-96">{JSON.stringify(anomaly.details, null, 2)}</pre>
            </TerminalCard>
          </TabPanel>
          <TabPanel value="fixes">
            <SuggestedFixes anomalyId={anomaly.id} />
          </TabPanel>
          <TabPanel value="chart">
            <MetricChart runId={anomaly.run_id} metricName={anomaly.details.metric_name} />
          </TabPanel>
        </Tabs>
      </div>
    </div>
  )
}