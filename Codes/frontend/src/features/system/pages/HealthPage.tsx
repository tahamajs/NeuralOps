import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function HealthPage() {
  const { data } = useQuery({
    queryKey: ['health'],
    queryFn: async () => (await api.get('/api/health/')).data,
  })
  if (!data) return <Spinner />
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">SYSTEM <span className="text-neon-blue">HEALTH</span></h1>
      <Card>
        <p className="text-sm text-slate-400">Status: <span className="text-green-400 font-bold uppercase">{data.status}</span></p>
        <p className="text-sm text-slate-400">Database: {data.database}</p>
        <p className="text-sm text-slate-400">Timestamp: {data.timestamp}</p>
      </Card>
    </div>
  )
}