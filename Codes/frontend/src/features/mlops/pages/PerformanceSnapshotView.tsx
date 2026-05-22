// src/features/mlops/pages/PerformanceSnapshotView.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { mlopsAPI } from '../services/api'
import { SnapshotTable } from '../components/SnapshotTable'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts'
import { Activity, Camera, Filter } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function PerformanceSnapshotView({ onClose }: { onClose?: () => void }) {
  const [selectedModel, setSelectedModel] = useState('')
  const [timeRange, setTimeRange] = useState('30d')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: snapshots, isLoading } = useQuery({
    queryKey: ['performance-snapshots', selectedModel, timeRange],
    queryFn: () => mlopsAPI.getPerformanceSnapshots({ model_id: selectedModel || undefined }),
  })

  const triggerSnapshot = useMutation({
    mutationFn: (modelId: string) => mlopsAPI.triggerSnapshot(modelId),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['performance-snapshots'] }); success('Snapshot triggered') },
  })

  // Prepare chart data
  const chartData = snapshots?.map(s => ({
    date: new Date(s.timestamp).toLocaleDateString(),
    accuracy: s.metrics.accuracy,
    precision: s.metrics.precision,
    recall: s.metrics.recall,
    drift: s.data_drift_score,
  })) || []

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Activity size={20} className="text-neon-mint" /> Performance Snapshots</h1><p className="text-xs text-slate-400">Track model performance over time</p></div>
        <div className="flex gap-2">{selectedModel && <Button size="sm" onClick={() => triggerSnapshot.mutate(selectedModel)}><Camera size={14} className="mr-1" /> Capture Snapshot</Button>}{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-6">
        <div className="flex gap-2"><Input placeholder="Model ID" value={selectedModel} onChange={e => setSelectedModel(e.target.value)} className="w-64" /><Select options={[{ value: '7d', label: 'Last 7 days' }, { value: '30d', label: 'Last 30 days' }, { value: '90d', label: 'Last 90 days' }]} value={timeRange} onChange={e => setTimeRange(e.target.value)} className="w-32" /></div>
        {snapshots && snapshots.length > 0 ? (<>
          <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Metric Trends</h3><ResponsiveContainer width="100%" height={300}><LineChart data={chartData}><XAxis dataKey="date" stroke="#475569" /><YAxis stroke="#475569" /><Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} /><Legend /><Line type="monotone" dataKey="accuracy" stroke="#3b82f6" strokeWidth={2} /><Line type="monotone" dataKey="precision" stroke="#10b981" strokeWidth={2} /><Line type="monotone" dataKey="recall" stroke="#f59e0b" strokeWidth={2} /><Line type="monotone" dataKey="drift" stroke="#ef4444" strokeWidth={2} strokeDasharray="5 5" /></LineChart></ResponsiveContainer></TerminalCard>
          <TerminalCard className="p-4"><SnapshotTable snapshots={snapshots} /></TerminalCard>
        </>) : <div className="text-center text-slate-500 py-12">No snapshots found. Select a model to view or trigger a new snapshot.</div>}
      </div>
    </div>
  )
}