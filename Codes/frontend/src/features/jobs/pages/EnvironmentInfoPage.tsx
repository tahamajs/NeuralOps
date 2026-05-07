import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function EnvironmentInfoPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['environment', jobId],
    queryFn: async () => (await api.get(`/api/jobs/${jobId}/environment/`)).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        ENVIRONMENT <span className="text-neon-blue">INFO</span>
      </h1>
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="text-slate-500">Python Version: </span>
            <span className="text-white">{data.python_version || 'N/A'}</span>
          </div>
          <div>
            <span className="text-slate-500">Git Commit: </span>
            <span className="text-white font-mono">{data.git?.commit?.slice(0, 7) || 'N/A'}</span>
          </div>
          <div>
            <span className="text-slate-500">Branch: </span>
            <span className="text-white">{data.git?.branch || 'N/A'}</span>
          </div>
          <div>
            <span className="text-slate-500">Repository: </span>
            <span className="text-white">{data.git?.repository || 'N/A'}</span>
          </div>
          <div>
            <span className="text-slate-500">Dirty: </span>
            <span className="text-white">{data.git?.dirty ? 'Yes' : 'No'}</span>
          </div>
          <div>
            <span className="text-slate-500">Data Commit: </span>
            <span className="text-white font-mono">{data.data_version?.git_commit?.slice(0, 7) || 'N/A'}</span>
          </div>
          <div>
            <span className="text-slate-500">DVC Hash: </span>
            <span className="text-white font-mono">{data.data_version?.dvc_hash?.slice(0, 12) || 'N/A'}</span>
          </div>
        </div>
        {data.pip_freeze && (
          <details className="mt-4">
            <summary className="text-neon-blue cursor-pointer text-sm">Pip Freeze</summary>
            <pre className="bg-cyber-800 p-3 rounded-lg text-xs text-slate-400 max-h-48 overflow-auto mt-2">
              {data.pip_freeze}
            </pre>
          </details>
        )}
      </Card>
    </div>
  )
}