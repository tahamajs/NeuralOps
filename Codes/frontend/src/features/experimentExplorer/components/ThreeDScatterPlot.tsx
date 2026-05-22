// src/features/experimentExplorer/components/ThreeDScatterPlot.tsx
import Plot from 'react-plotly.js'
import { ExperimentRun } from '../types'

export function ThreeDScatterPlot({ experiments, xParam, yParam, zParam, colorMetric }: any) {
  const colors = experiments.map(exp => exp.metrics[colorMetric])

  return (
    <Plot
      data={[{
        type: 'scatter3d',
        mode: 'markers',
        x: experiments.map(exp => exp.parameters[xParam]),
        y: experiments.map(exp => exp.parameters[yParam]),
        z: experiments.map(exp => exp.parameters[zParam]),
        marker: { size: 5, color: colors, colorscale: 'Viridis', showscale: true, colorbar: { title: colorMetric } },
        text: experiments.map(exp => `Run: ${exp.name}<br>${xParam}: ${exp.parameters[xParam]}<br>${yParam}: ${exp.parameters[yParam]}<br>${zParam}: ${exp.parameters[zParam]}<br>${colorMetric}: ${exp.metrics[colorMetric]}`),
        hoverinfo: 'text',
      }]}
      layout={{
        autosize: true,
        height: 480,
        paper_bgcolor: 'rgba(0,0,0,0)',
        scene: {
          xaxis: { title: xParam, color: '#94a3b8' },
          yaxis: { title: yParam, color: '#94a3b8' },
          zaxis: { title: zParam, color: '#94a3b8' },
          bgcolor: 'rgba(0,0,0,0)',
        },
        margin: { t: 60, r: 60, b: 40, l: 60 },
        font: { color: '#94a3b8' },
      }}
      useResizeHandler
      style={{ width: '100%', height: '100%' }}
    />
  )
}