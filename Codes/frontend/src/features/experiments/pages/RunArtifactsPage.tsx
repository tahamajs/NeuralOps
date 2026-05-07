import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { Button } from '../../../components/ui/Button'

export default function RunArtifactsPage() {
  const { runId } = useParams<{ runId: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['runArtifacts', runId],
    queryFn: async () => (await api.get(`/api/runs/${runId}/artifacts/`)).data,
  })

  if (isLoading) return <Spinner />
  const files = data?.files || {}

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">ARTIFACTS</h1>
      <Card>
        {Object.keys(files).length === 0 ? (
          <p className="text-slate-500">No artifacts found.</p>
        ) : (
          <div className="space-y-2">
            {Object.entries(files).map(([name, url]: any) => (
              <div key={name} className="flex justify-between items-center bg-cyber-800 p-3 rounded">
                <span className="text-sm text-white">{name}</span>
                <a href={url} download>
                  <Button size="sm" variant="secondary">Download</Button>
                </a>
              </div>
            ))}
          </div>
        )}
      </Card>
    </div>
  )
}