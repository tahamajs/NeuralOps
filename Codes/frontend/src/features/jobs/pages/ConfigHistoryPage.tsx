import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'

export default function ConfigHistoryPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['configHistory', jobId],
    queryFn: async () => (await api.get(`/api/jobs/${jobId}/config/history/`)).data,
  })

  if (isLoading) return <Spinner />
  const versions = data?.versions || []

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        CONFIG <span className="text-neon-blue">HISTORY</span>
      </h1>
      {versions.length === 0 ? (
        <p className="text-slate-500">No versions yet.</p>
      ) : (
        versions.map((v: any) => (
          <Card key={v.version} hover className="space-y-2">
            <div className="flex justify-between items-center">
              <Badge status="completed">Version {v.version}</Badge>
              <span className="text-xs text-slate-400">{new Date(v.created_at).toLocaleString()}</span>
            </div>
            <p className="text-slate-500 text-xs">{v.description}</p>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 max-h-48 overflow-auto">
              {JSON.stringify(v.snapshot, null, 2)}
            </pre>
          </Card>
        ))
      )}
    </div>
  )
}