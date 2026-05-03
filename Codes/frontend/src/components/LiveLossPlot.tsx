import Plot from 'react-plotly.js'

export default function LiveLossPlot({ metrics }: { metrics: { loss: number[]; accuracy: number[] } }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Training Loss</h2>
      <Plot
        data={[
          {
            x: metrics.loss.map((_, i) => i + 1),
            y: metrics.loss,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Loss',
          },
        ]}
        layout={{
          autosize: true,
          margin: { t: 10, r: 10, b: 30, l: 40 },
          paper_bgcolor: 'transparent',
          plot_bgcolor: 'transparent',
          height: 300,
        }}
        useResizeHandler
        style={{ width: '100%' }}
      />
    </div>
  )
}