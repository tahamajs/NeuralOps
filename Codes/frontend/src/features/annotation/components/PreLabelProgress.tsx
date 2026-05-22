import { Progress } from '../../../components/ui/Progress'

export function PreLabelProgress({ progress, status }: any) {
  if (!status) return null
  return <TerminalCard className="p-4"><div className="flex justify-between items-center"><span className="text-sm text-white">Pre‑labeling progress</span><span className="text-xs text-slate-400">{status.processed} / {status.total}</span></div><Progress value={progress || (status.processed / status.total) * 100} className="h-1 mt-2" /></TerminalCard>
}