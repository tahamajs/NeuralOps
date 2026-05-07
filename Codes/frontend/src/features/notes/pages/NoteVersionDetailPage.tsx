import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function NoteVersionDetailPage() {
  const { noteId, versionNumber } = useParams<{ noteId: string; versionNumber: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['noteVersion', noteId, versionNumber],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/versions/${versionNumber}/`)).data,
  })

  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        VERSION <span className="text-neon-blue">{versionNumber}</span>
      </h1>
      <Card>
        <h3 className="text-white font-semibold">{data.title}</h3>
        <div className="text-xs text-slate-400 mt-1">
          Changed by {data.changed_by || 'unknown'} · {new Date(data.created_at).toLocaleString()}
        </div>
        <pre className="mt-4 bg-cyber-800 p-4 rounded text-slate-300 whitespace-pre-wrap">{data.content}</pre>
      </Card>
    </div>
  )
}