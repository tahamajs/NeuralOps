import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Link } from 'react-router-dom'

export default function NoteVersionsPage() {
  const { noteId } = useParams<{ noteId: string }>()
  const { data: versions = [], isLoading } = useQuery({
    queryKey: ['noteVersions', noteId],
    queryFn: async () => (await api.get(`/api/notes/v1/notes/${noteId}/versions/`)).data,
  })

  if (isLoading) return <Spinner/>
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">NOTE <span className="text-neon-blue">VERSIONS</span></h1>
      {versions.length === 0 ? (
        <p className="text-slate-500">No previous versions.</p>
      ) : (
        versions.map((v: any) => (
          <Card key={v.version} hover>
            <div className="flex justify-between items-center">
              <span className="text-neon-blue font-bold">Version {v.version}</span>
              <span className="text-xs text-slate-400">{new Date(v.created_at).toLocaleString()}</span>
            </div>
            <p className="text-xs text-slate-500 mt-1">{v.change_reason}</p>
            <pre className="bg-cyber-800 p-3 rounded mt-2 text-xs text-slate-400 max-h-32 overflow-auto">
              {v.content_preview}
            </pre>
            <Link to={`/notes/${noteId}/versions/${v.version}`} className="text-neon-blue text-xs mt-2 inline-block">
              View Full Version
            </Link>
          </Card>
        ))
      )}
    </div>
  )
}