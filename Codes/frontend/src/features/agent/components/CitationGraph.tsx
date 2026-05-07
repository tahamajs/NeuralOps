import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import Plot from 'react-plotly.js'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

interface Props {
  paperIds: string[]
}

export default function CitationGraph({ paperIds }: Props) {
  const { data, isLoading } = useQuery({
    queryKey: ['citationGraph', paperIds],
    queryFn: async () => (await api.post('/api/agents/papers/citation-graph/', { paper_ids: paperIds })).data,
    enabled: paperIds.length > 0,
  })

  if (isLoading) return <Spinner />
  if (!data?.nodes?.length) return null

  // Build Plotly network graph
  const nodeTrace = {
    x: data.nodes.map((_: any, i: number) => Math.cos(2 * Math.PI * i / data.nodes.length) * 100),
    y: data.nodes.map((_: any, i: number) => Math.sin(2 * Math.PI * i / data.nodes.length) * 100),
    mode: 'markers+text' as const,
    type: 'scatter' as const,
    text: data.nodes.map((n: any) => n.title.slice(0, 30)),
    textposition: 'top center' as const,
    marker: {
      size: data.nodes.map((n: any) => Math.max(10, Math.min(40, (n.citation_count || 1) / 10))),
      color: data.nodes.map((n: any) => n.year || 2020),
      colorscale: 'Viridis',
      showscale: true,
      colorbar: { title: 'Year' },
    },
    hoverinfo: 'text' as const,
    hovertext: data.nodes.map((n: any) => `${n.title}<br>Year: ${n.year}<br>Citations: ${n.citation_count}`),
  }

  const edgeTraces = data.edges.map((e: any) => {
    const fromIdx = data.nodes.findIndex((n: any) => n.id === e.from)
    const toIdx = data.nodes.findIndex((n: any) => n.id === e.to)
    if (fromIdx < 0 || toIdx < 0) return null
    return {
      x: [Math.cos(2 * Math.PI * fromIdx / data.nodes.length) * 100, Math.cos(2 * Math.PI * toIdx / data.nodes.length) * 100],
      y: [Math.sin(2 * Math.PI * fromIdx / data.nodes.length) * 100, Math.sin(2 * Math.PI * toIdx / data.nodes.length) * 100],
      mode: 'lines' as const,
      type: 'scatter' as const,
      line: { color: '#ffffff30', width: 1 },
      hoverinfo: 'none' as const,
      showlegend: false,
    }
  }).filter(Boolean)

  return (
    <Card>
      <h3 className="text-sm font-bold text-white mb-3">Citation Graph</h3>
      <Plot
        data={[nodeTrace, ...edgeTraces]}
        layout={{
          autosize: true,
          height: 400,
          showlegend: false,
          hovermode: 'closest',
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          font: { color: '#94a3b8' },
          xaxis: { visible: false },
          yaxis: { visible: false },
          margin: { t: 10, r: 10, b: 10, l: 10 },
        }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    </Card>
  )
}