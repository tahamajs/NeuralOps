import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import Plot from 'react-plotly.js'
import { Button } from '../../../components/ui/Button'
import { X, Table2, BarChart3 } from 'lucide-react'

interface Props {
  runIds: string[]
  onClose: () => void
}

export default function CompareDrawer({ runIds, onClose }: Props) {
  const [view, setView] = useState<'config' | 'metrics'>('config')
  const { data, isLoading } = useQuery({
    queryKey: ['compare', runIds],
    queryFn: () => api.get('/runs/compare/', { params: { run_ids: runIds.join(',') } }).then(r => r.data),
    enabled: runIds.length >= 2,
  })

  return (
    <div className="fixed inset-y-0 right-0 w-[420px] glass border-l border-white/10 z-50 p-6 overflow-y-auto animate-scale-in">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-lg font-display font-bold text-white">Compare {runIds.length} Runs</h2>
        <button onClick={onClose}><X size={20} className="text-slate-400 hover:text-white" /></button>
      </div>
      <div className="flex gap-2 mb-4">
        <Button size="sm" variant={view==='config'?'primary':'ghost'} onClick={() => setView('config')}>
          <Table2 size={14} className="mr-1" /> Config
        </Button>
        <Button size="sm" variant={view==='metrics'?'primary':'ghost'} onClick={() => setView('metrics')}>
          <BarChart3 size={14} className="mr-1" /> Metrics
        </Button>
      </div>
      {isLoading ? (
        <div className="text-slate-400 text-sm text-center py-8">Loading...</div>
      ) : !data ? (
        <div className="text-slate-400 text-sm text-center py-8">No data</div>
      ) : view === 'config' ? (
        <table className="w-full text-xs">
          <thead>
            <tr>
              <th className="text-left py-1 text-slate-400">Key</th>
              {Object.keys(data.config_diff.runs).map((rid: string) => (
                <th key={rid} className="text-left py-1 text-white font-mono text-[10px]">{rid.slice(0,6)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.config_diff.keys.map((key: string) => (
              <tr key={key} className="border-b border-white/5">
                <td className="py-1 text-slate-400">{key}</td>
                {Object.entries(data.config_diff.runs).map(([rid, cfg]: any) => (
                  <td key={rid} className="py-1 text-white">{JSON.stringify(cfg[key])}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <Plot
          data={Object.entries(data.metrics.loss).map(([rid, pts]: any) => ({
            x: pts.map((p: any) => p[0]),
            y: pts.map((p: any) => p[1]),
            type: 'scatter',
            mode: 'lines',
            name: rid.slice(0,6),
          }))}
          layout={{ autosize: true, height: 400, margin: { t: 20, r: 20, b: 40, l: 50 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#94a3b8' } }}
          useResizeHandler
          style={{ width: '100%' }}
        />
      )}
    </div>
  )
}