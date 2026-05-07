import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function HPOStatusPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['hpoStatus'],
    queryFn: async () => (await api.get('/api/hpo/status/')).data,
    refetchInterval: 5000,
  })

  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">HPO Status</h1>
      <Card>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">Status</div>
            <div className="text-xl font-bold text-white">{data.status}</div>
          </div>
          <div>
            <div className="text-xs text-slate-500 uppercase tracking-widest">Trials Completed</div>
            <div className="text-xl font-bold text-white">{data.trials_completed}</div>
          </div>
        </div>
      </Card>
    </div>
  )
}