import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Avatar } from '../../../components/ui'

export default function PublicPortfolioPage() {
  const { username } = useParams()
  const { data, isLoading } = useQuery({
    queryKey: ['portfolio', username],
    queryFn: async () => (await api.get(`/portfolios/user/${username}/`)).data,
  })
  if (isLoading) return <Spinner />
  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <Card className="text-center">
        <Avatar src={data?.avatar_url} size="lg" />
        <h1 className="text-2xl font-bold text-white">{data?.username}</h1>
        <p className="text-slate-400">{data?.bio}</p>
      </Card>
      <Card>
        <h2 className="text-sm font-bold text-white mb-2">Showcase Items</h2>
        <div className="grid gap-4">
          {data?.showcase_items?.map((item: any) => (
            <div key={item.id} className="border-b border-white/5 py-2">
              <h3 className="font-semibold text-white">{item.title}</h3>
              <p className="text-sm text-slate-300">{item.description}</p>
            </div>
          ))}
        </div>
      </Card>
    </div>
  )
}