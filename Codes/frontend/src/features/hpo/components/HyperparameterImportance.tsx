import { useQuery } from '@tanstack/react-query'
import { hpoAPI } from '../services/api'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts'

export function HyperparameterImportance({ sweepId }: { sweepId: string }) {
  const { data } = useQuery({
    queryKey: ['param-importance', sweepId],
    queryFn: () => hpoAPI.getParameterImportance(sweepId),
  })
  if (!data?.length) return <div className="text-slate-500 text-center py-8">Not enough trials for importance analysis</div>
  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={data} layout="vertical" margin={{ left: 80 }}>
        <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
        <YAxis type="category" dataKey="name" tick={{ fill: '#94a3b8' }} />
        <Tooltip contentStyle={{ backgroundColor: '#0a0f1a' }} />
        <Bar dataKey="importance" fill="#3b82f6" radius={[0, 4, 4, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}