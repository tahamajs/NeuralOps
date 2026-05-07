import { useState, useEffect, useCallback } from 'react'
import { useQuery } from '@tanstack/react-query'
import Plot from 'react-plotly.js'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'

interface MetricGroup {
  steps: number[]
  values: any[]
  type: 'scalar' | 'image' | 'table'
}

interface MetricsData {
  [name: string]: MetricGroup
}

interface CustomMetricsDashboardProps {
  runId: string
}

export default function CustomMetricsDashboard({ runId }: CustomMetricsDashboardProps) {
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([])
  const [autoRefresh, setAutoRefresh] = useState(true)
  const [fullImage, setFullImage] = useState<{ src: string; title: string } | null>(null)

  const { data: metrics, isLoading, isError, error, refetch } = useQuery<MetricsData>({
    queryKey: ['customMetrics', runId],
    queryFn: async () => {
      const res = await api.get(`/api/runs/${runId}/metrics/`)
      return res.data
    },
    enabled: !!runId,
    refetchInterval: autoRefresh ? 10000 : false,
  })

  // Auto-refresh toggle doesn't need effect; it's handled by `refetchInterval`

  const scalarMetrics = metrics
    ? Object.entries(metrics).filter(([_, group]) => group.type === 'scalar')
    : []

  const imageMetrics = metrics
    ? Object.entries(metrics).filter(([_, group]) => group.type === 'image')
    : []

  const tableMetrics = metrics
    ? Object.entries(metrics).filter(([_, group]) => group.type === 'table')
    : []

  const handleMetricToggle = (name: string) => {
    setSelectedMetrics(prev =>
      prev.includes(name) ? prev.filter(n => n !== name) : [...prev, name]
    )
  }

  // Build Plotly traces
  const chartData = selectedMetrics
    .filter(name => metrics?.[name]?.type === 'scalar')
    .map(name => ({
      x: metrics![name].steps,
      y: metrics![name].values,
      type: 'scatter' as const,
      mode: 'lines+markers' as const,
      name,
      line: { width: 2 },
      marker: { size: 4 },
    }))

  if (isLoading) return <Spinner className="py-8" />
  if (isError) return <p className="text-red-400 text-sm">Failed to load metrics: {(error as any)?.message}</p>
  if (!metrics || Object.keys(metrics).length === 0) {
    return (
      <Card>
        <h3 className="text-sm font-bold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-purple-500 rounded-full" />
          Custom Metrics Dashboard
        </h3>
        <p className="text-slate-500 text-sm">No custom metrics have been logged for this run yet.</p>
      </Card>
    )
  }

  return (
    <Card>
      <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
        <h3 className="text-sm font-bold text-white flex items-center gap-2">
          <span className="w-1 h-4 bg-purple-500 rounded-full" />
          Custom Metrics Dashboard
        </h3>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant={autoRefresh ? 'secondary' : 'ghost'}
            onClick={() => setAutoRefresh(!autoRefresh)}
          >
            Auto {autoRefresh ? 'On' : 'Off'}
          </Button>
          <Button size="sm" variant="ghost" onClick={() => refetch()}>
            Refresh Now
          </Button>
        </div>
      </div>

      {/* Scalar selection + plot */}
      {scalarMetrics.length > 0 && (
        <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
          <div className="lg:col-span-1">
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-2">
              Select Metrics
            </label>
            <div className="max-h-48 overflow-y-auto scrollbar-thin border border-white/10 rounded-lg">
              {scalarMetrics.map(([name]) => (
                <label
                  key={name}
                  className={`flex items-center gap-2 px-3 py-2 cursor-pointer transition-colors ${
                    selectedMetrics.includes(name)
                      ? 'bg-neon-blue/10 text-neon-blue'
                      : 'hover:bg-white/5 text-slate-400'
                  }`}
                >
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(name)}
                    onChange={() => handleMetricToggle(name)}
                    className="accent-neon-blue"
                  />
                  <span className="text-xs truncate">{name}</span>
                </label>
              ))}
            </div>
            <Button
              size="sm"
              onClick={() => {}}
              disabled={selectedMetrics.length === 0}
              className="w-full mt-2"
            >
              Update Chart
            </Button>
          </div>
          <div className="lg:col-span-3">
            {chartData.length > 0 ? (
              <Plot
                data={chartData}
                layout={{
                  autosize: true,
                  height: 350,
                  margin: { t: 20, r: 20, b: 40, l: 50 },
                  paper_bgcolor: 'transparent',
                  plot_bgcolor: 'transparent',
                  font: { color: '#94a3b8' },
                  legend: { x: 1, y: 1, xanchor: 'right', bgcolor: 'rgba(0,0,0,0.5)' },
                  hovermode: 'closest',
                }}
                useResizeHandler
                style={{ width: '100%' }}
              />
            ) : (
              <div className="flex items-center justify-center h-[350px] text-slate-500 text-sm">
                Select one or more metrics to plot
              </div>
            )}
          </div>
        </div>
      )}

      {/* Image Gallery */}
      {imageMetrics.length > 0 && (
        <div className="mb-6">
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Logged Images</h4>
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-3">
            {imageMetrics.map(([name, group]) => {
              // Show the last image for each metric
              const lastImg = group.values[group.values.length - 1]
              const lastStep = group.steps[group.steps.length - 1]
              return (
                <div
                  key={name}
                  className="bg-cyber-800 rounded-lg overflow-hidden cursor-pointer hover:ring-2 ring-neon-blue transition-all"
                  onClick={() =>
                    setFullImage({
                      src: lastImg,
                      title: `${name} (step ${lastStep})`,
                    })
                  }
                >
                  <img
                    src={lastImg}
                    alt={`${name} step ${lastStep}`}
                    className="w-full h-32 object-contain bg-black/50"
                  />
                  <div className="p-2 text-[10px] text-slate-400 truncate">
                    {name} (step {lastStep})
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Tables */}
      {tableMetrics.length > 0 && (
        <div>
          <h4 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-3">Logged Tables</h4>
          {tableMetrics.map(([name, group]) => {
            const lastTable = group.values[group.values.length - 1]
            const headers = Object.keys(lastTable)
            return (
              <div key={name} className="mb-4">
                <h5 className="text-sm font-semibold text-white mb-2">{name}</h5>
                <div className="overflow-x-auto">
                  <table className="w-full text-xs border border-white/10 rounded-lg overflow-hidden">
                    <thead>
                      <tr className="bg-cyber-700">
                        {headers.map(h => (
                          <th key={h} className="px-3 py-2 text-left text-slate-400 font-medium">
                            {h}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      <tr className="border-t border-white/5">
                        {headers.map(h => (
                          <td key={h} className="px-3 py-2 text-white">
                            {String(lastTable[h])}
                          </td>
                        ))}
                      </tr>
                    </tbody>
                  </table>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div className="text-[10px] text-slate-500 mt-4 text-center">
        {autoRefresh ? 'Auto-refreshing every 10s' : 'Auto-refresh off'} · Last updated: {new Date().toLocaleTimeString()}
      </div>

      {/* Full-screen image modal */}
      {fullImage && (
        <div
          className="fixed inset-0 z-50 bg-black/95 flex items-center justify-center cursor-pointer"
          onClick={() => setFullImage(null)}
        >
          <div className="max-w-[90%] max-h-[90%]">
            <img
              src={fullImage.src}
              alt={fullImage.title}
              className="max-w-full max-h-[90vh] rounded-lg shadow-2xl"
            />
            <p className="text-center mt-3 text-slate-400 text-sm">{fullImage.title}</p>
          </div>
        </div>
      )}
    </Card>
  )
}