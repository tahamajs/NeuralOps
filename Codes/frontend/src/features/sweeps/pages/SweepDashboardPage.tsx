import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'
import Plot from 'react-plotly.js'
import { Badge } from '../../../components/ui/Badge'

export default function SweepDashboardPage() {
  const { sweepId } = useParams<{ sweepId: string }>()
  const { data: sweep, isLoading } = useQuery({
    queryKey: ['sweep', sweepId],
    queryFn: async () => (await api.get(`/api/sweeps/${sweepId}/`)).data,
    refetchInterval: 5000,
  })

  const { data: importance } = useQuery({
    queryKey: ['sweepImportance', sweepId],
    queryFn: async () => (await api.get(`/api/sweeps/${sweepId}/importance/`)).data,
    enabled: !!sweep,
  })

  if (isLoading || !sweep) return <Spinner />

  const trials = sweep.trials || []
  const bestTrial = trials.filter((t: any) => t.status === 'completed').sort((a: any, b: any) => b.objective_value - a.objective_value)[0]

  // Parallel coordinates data
  const paramNames = Object.keys(trials[0]?.params || {})
  const pcData = paramNames.map((param) => ({
    type: 'parcoords' as const,
    line: {
      color: trials.map((t: any) => t.objective_value || 0),
      colorscale: 'Jet',
      showscale: true,
      cmin: Math.min(...trials.map((t: any) => t.objective_value || 0)),
      cmax: Math.max(...trials.map((t: any) => t.objective_value || 1)),
    },
    dimensions: [{
      range: [0, Math.max(...trials.map((t: any) => t.objective_value || 0))],
      label: 'Objective',
      values: trials.map((t: any) => t.objective_value || 0),
    }, ...paramNames.map((param) => {
      const values = trials.map((t: any) => t.params[param])
      const isNum = values.every((v: any) => typeof v === 'number')
      return isNum
        ? { range: [Math.min(...values), Math.max(...values)], label: param, values }
        : { tickvals: [...new Set(values)], ticktext: [...new Set(values)], label: param, values }
    })],
  }))

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-2xl font-display font-bold text-white">{sweep.name}</h1>
          <p className="text-slate-400 text-sm">{sweep.trials.length} trials · Status: {sweep.status}</p>
        </div>
        <Button size="sm" onClick={() => api.post(`/api/sweeps/${sweepId}/`, { status: sweep.status === 'active' ? 'paused' : 'active' })}>
          {sweep.status === 'active' ? 'Pause' : 'Resume'}
        </Button>
      </div>

      {bestTrial && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-2">Best Trial</h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
            <div><span className="text-slate-400">Trial #</span> <span className="text-white">#{bestTrial.trial_number}</span></div>
            <div><span className="text-slate-400">Objective</span> <span className="text-green-400 font-bold">{bestTrial.objective_value?.toFixed(4)}</span></div>
            <div><span className="text-slate-400">Status</span> <Badge status={bestTrial.status}>{bestTrial.status}</Badge></div>
            <div><span className="text-slate-400">Params</span> <span className="text-white text-[10px]">{JSON.stringify(bestTrial.params)}</span></div>
          </div>
        </Card>
      )}

      {trials.length > 1 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Parallel Coordinates</h3>
          <Plot data={pcData} layout={{ autosize: true, height: 400, margin: { t: 20, r: 20, b: 40, l: 60 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#94a3b8' } }} useResizeHandler style={{ width: '100%' }} />
        </Card>
      )}

      {importance && Object.keys(importance).length > 0 && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Parameter Importance</h3>
          <Plot
            data={[{ x: Object.values(importance), y: Object.keys(importance), type: 'bar', orientation: 'h', marker: { color: '#3b82f6' } }]}
            layout={{ autosize: true, height: 200 + Object.keys(importance).length * 20, margin: { t: 10, r: 20, b: 30, l: 100 }, paper_bgcolor: 'transparent', plot_bgcolor: 'transparent', font: { color: '#94a3b8' } }}
            useResizeHandler
            style={{ width: '100%' }}
          />
        </Card>
      )}

      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Trials</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead>
              <tr className="text-left text-slate-400">
                <th className="py-2 px-2">#</th>
                <th className="py-2 px-2">Status</th>
                <th className="py-2 px-2">Objective</th>
                <th className="py-2 px-2">Params</th>
                <th className="py-2 px-2">Created</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {trials.map((trial: any) => (
                <tr key={trial.id}>
                  <td className="py-2 px-2">{trial.trial_number}</td>
                  <td className="py-2 px-2"><Badge status={trial.status}>{trial.status}</Badge></td>
                  <td className="py-2 px-2">{trial.objective_value?.toFixed(4) || '—'}</td>
                  <td className="py-2 px-2 text-[10px]">{JSON.stringify(trial.params)}</td>
                  <td className="py-2 px-2 text-slate-500">{new Date(trial.created_at).toLocaleString()}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}