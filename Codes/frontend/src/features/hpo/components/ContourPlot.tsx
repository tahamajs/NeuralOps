import { useQuery } from '@tanstack/react-query'
import { hpoAPI } from '../services/api'
import Plot from 'react-plotly.js'

export function ContourPlot({ sweepId }: { sweepId: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['contour', sweepId],
    queryFn: () => hpoAPI.getContourData(sweepId),
  })
  if (isLoading) return <div className="text-slate-500 text-center py-8">Loading contour plot...</div>
  if (!data) return <div className="text-slate-500 text-center py-8">Not enough data for contour plot</div>
  return <Plot data={[data]} layout={{ autosize: true, height: 500, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)', font: { color: '#94a3b8' } }} useResizeHandler style={{ width: '100%' }} />
}