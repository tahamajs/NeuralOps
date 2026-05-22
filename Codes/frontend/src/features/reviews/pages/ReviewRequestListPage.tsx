import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Badge, Spinner, Link } from '../../../components/ui'
import { Plus } from 'lucide-react'

export default function ReviewRequestListPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['review-requests'],
    queryFn: async () => (await api.get('/requests/')).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <div className="flex justify-between">
        <h1 className="text-2xl font-bold text-white">Peer Review Requests</h1>
        <Link to="/reviews/new"><Button size="sm"><Plus size={14} /> Request Review</Button></Link>
      </div>
      <div className="grid gap-4">
        {data?.results?.map((req: any) => (
          <Card key={req.id} hover>
            <div className="flex justify-between">
              <div>
                <h3 className="font-bold text-white">{req.title}</h3>
                <p className="text-xs text-slate-400">By {req.requester_username} · {new Date(req.created_at).toLocaleDateString()}</p>
                <div className="flex gap-2 mt-2">
                  {req.tags?.map((t: string) => <Badge key={t}>{t}</Badge>)}
                </div>
              </div>
              <Badge variant={req.status}>{req.status}</Badge>
            </div>
            <Link to={`/reviews/${req.id}`} className="mt-3 inline-block"><Button size="sm" variant="ghost">View Details →</Button></Link>
          </Card>
        ))}
      </div>
    </div>
  )
}