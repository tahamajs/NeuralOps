import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'

export default function ActivityFeedPage() {
  const { data: activities = [], isLoading } = useQuery({
    queryKey: ['activity'],
    queryFn: async () => (await api.get('/api/activity/')).data,
    refetchInterval: 10000,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        ACTIVITY <span className="text-neon-blue">FEED</span>
      </h1>
      <div className="space-y-3">
        {activities.map((a: any) => (
          <Card key={a.id} hover className="flex items-center gap-4">
            <div className="w-8 h-8 rounded-full bg-neon-blue/20 flex items-center justify-center text-neon-blue font-bold text-xs">
              {a.username?.[0]?.toUpperCase() || 'S'}
            </div>
            <div className="flex-1">
              <p className="text-sm text-white">
                <span className="font-semibold">{a.username || 'system'}</span>{' '}
                <span className="text-slate-400">{a.description}</span>
              </p>
              <p className="text-[10px] text-slate-500 mt-0.5">{new Date(a.created_at).toLocaleString()}</p>
            </div>
            <Badge status={a.action_type}>{a.action_type}</Badge>
          </Card>
        ))}
      </div>
    </div>
  )
}