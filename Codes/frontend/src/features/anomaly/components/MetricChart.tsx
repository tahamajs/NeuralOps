import { useQuery } from '@tanstack/react-query'
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Spinner } from '../../../components/ui/Spinner'
import api from '../../../api/client'

export function MetricChart({ runId, metricName }: { runId: string; metricName?: string }) {
  const { data, isLoading } = useQuery({
    queryKey: ['run-metrics', runId, metricName],
    queryFn: () => api.get(`/runs/${runId}/history/?metric=${metricName || 'loss'}`).then(res => res.data),
  })
  if (isLoading) return <Spinner size="sm" />
  if (!data?.length) return <div className="text-slate-500">No metric data available</div>
  return (
    <ResponsiveContainer width="100%" height={300}>
      <LineChart data={data}>
        <XAxis dataKey="step" stroke="#475569" />
        <YAxis stroke="#475569" />
        <Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} />
        <Line type="monotone" dataKey="value" stroke="#3b82f6" strokeWidth={2} dot={false} />
        <ReferenceLine y={data[0]?.value * 1.2} stroke="#ef4444" strokeDasharray="3 3" label="Anomaly threshold" />
      </LineChart>
    </ResponsiveContainer>
  )
}