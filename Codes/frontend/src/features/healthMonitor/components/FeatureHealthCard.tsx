import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import { Progress } from '../../../components/ui/Progress'
import { DriftMonitor } from '../types'

export function FeatureHealthCard({ monitor }: { monitor: DriftMonitor }) {
  return <TerminalCard className="p-2"><div className="flex justify-between items-center"><span className="font-mono text-sm">{monitor.feature_name}</span><Badge variant={monitor.status === 'normal' ? 'success' : monitor.status === 'warning' ? 'warning' : 'danger'}>{monitor.status}</Badge></div><Progress value={monitor.status === 'normal' ? 30 : monitor.status === 'warning' ? 70 : 100} className="h-1 mt-1" /><div className="text-xs text-slate-400 mt-1">Threshold: {monitor.alert_threshold}</div></TerminalCard>
}