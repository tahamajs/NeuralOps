import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function RunAnalysisOverviewPage() {
  const { runId } = useParams<{ runId: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['runAnalysis', runId],
    queryFn: async () => (await api.get(`/api/runs/${runId}/analysis/`)).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        ANALYSIS <span className="text-neon-blue">OVERVIEW</span>
      </h1>
      <Card>
        <p className="text-slate-400 text-sm mb-4">Run ID: {runId}</p>
        <div className="grid grid-cols-2 gap-3">
          <Link to={`/runs/${runId}/drift`} className="block p-3 bg-cyber-800 rounded-lg text-white hover:bg-cyber-700">
            Drift Report {data.has_drift_report ? '✅' : '❌'}
          </Link>
          <Link to={`/runs/${runId}/fairness`} className="block p-3 bg-cyber-800 rounded-lg text-white hover:bg-cyber-700">
            Fairness Report {data.has_fairness_report ? '✅' : '❌'}
          </Link>
          <Link to={`/runs/${runId}/robustness`} className="block p-3 bg-cyber-800 rounded-lg text-white hover:bg-cyber-700">
            Robustness Report {data.has_robustness_report ? '✅' : '❌'}
          </Link>
          <Link to={`/runs/${runId}/artifacts`} className="block p-3 bg-cyber-800 rounded-lg text-white hover:bg-cyber-700">
            Artifacts
          </Link>
        </div>
      </Card>
    </div>
  )
}