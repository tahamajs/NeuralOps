import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Link } from 'react-router-dom'

export default function NoteRepliesPage() {
  const { noteId } = useParams<{ noteId: string }>()
  const { data: replies = [], isLoading } = useQuery({
    queryKey: ['noteReplies', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/replies/`)).data,
  })

  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        REPLIES <span className="text-neon-blue">({replies.length})</span>
      </h1>
      {replies.length === 0 ? (
        <p className="text-slate-500">No replies yet.</p>
      ) : (
        replies.map((reply: any) => (
          <Link key={reply.id} to={`/notes/${reply.id}`}>
            <Card hover>
              <div className="flex justify-between text-[10px] text-slate-500 mb-1">
                <span>{reply.author_username}</span>
                <span>{new Date(reply.created_at).toLocaleString()}</span>
              </div>
              <p className="text-sm text-white">{reply.content}</p>
            </Card>
          </Link>
        ))
      )}
    </div>
  )
}