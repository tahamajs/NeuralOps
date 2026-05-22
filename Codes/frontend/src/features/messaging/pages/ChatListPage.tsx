import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Link } from '../../../components/ui'

export default function ChatListPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['chats'],
    queryFn: async () => (await api.get('/chats/')).data,
  })
  if (isLoading) return <Spinner />
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold text-white">Messages</h1>
      {data?.results?.map((chat: any) => (
        <Link key={chat.id} to={`/chats/${chat.id}`}>
          <Card hover>
            <div className="flex justify-between">
              <div>
                <p className="font-bold text-white">{chat.participants.join(', ')}</p>
                <p className="text-sm text-slate-400 truncate">{chat.last_message}</p>
              </div>
              <span className="text-xs text-slate-500">{new Date(chat.updated_at).toLocaleDateString()}</span>
            </div>
          </Card>
        </Link>
      ))}
    </div>
  )
}