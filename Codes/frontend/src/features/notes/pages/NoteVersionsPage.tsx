import { useParams, Link } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { EmptyState } from '../../../components/ui/EmptyState'
import { History, ArrowLeft } from 'lucide-react'

export default function NoteVersionsPage() {
  const { noteId } = useParams<{ noteId: string }>()

  const { data: versions = [], isLoading } = useQuery({
    queryKey: ['noteVersions', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/versions/`)).data,
    enabled: !!noteId,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <div className="flex items-center gap-2">
        <Link to={`/notes/${noteId}`} className="text-slate-400 hover:text-white">
          <ArrowLeft size={18} />
        </Link>
        <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
          <History size={22} className="text-neon-blue" /> Version History
        </h1>
      </div>

      {versions.length === 0 ? (
        <EmptyState message="No previous versions yet." />
      ) : (
        <div className="space-y-3">
          {versions.map((v: any) => (
            <Card key={v.version} hover className="flex items-center justify-between">
              <div>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-neon-blue font-mono font-bold text-sm">Version {v.version}</span>
                  {v.change_reason && (
                    <span className="text-xs text-slate-400">· {v.change_reason}</span>
                  )}
                </div>
                <p className="text-xs text-slate-500">
                  Changed by {v.changed_by || 'unknown'} · {new Date(v.created_at).toLocaleString()}
                </p>
                <p className="text-xs text-slate-400 mt-1 line-clamp-2">{v.content_preview}</p>
              </div>
              <Link
                to={`/notes/${noteId}/versions/${v.version}`}
                className="text-neon-blue text-xs font-bold hover:underline"
              >
                View Full Version →
              </Link>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}