import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { BarChart3, TrendingUp } from 'lucide-react'

export default function AdvancedAnalyticsPage() {
  const { data: hpImportance, isLoading: hpLoading } = useQuery({
    queryKey: ['hpImportance'],
    queryFn: async () =>
      (await api.get('/api/experiments/analytics/hp-importance/')).data,
  })
  const { data: metricCorr, isLoading: corrLoading } = useQuery({
    queryKey: ['metricCorr'],
    queryFn: async () =>
      (await api.get('/api/experiments/analytics/metric-correlation/')).data,
  })

  if (hpLoading || corrLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-display font-bold text-white flex items-center gap-2">
        <BarChart3 size={24} className="text-neon-blue" />
        Advanced <span className="text-neon-blue">Analytics</span>
      </h1>

      {hpImportance && !hpImportance.error && (
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3 flex items-center gap-2">
            <TrendingUp size={16} className="text-neon-purple" />
            Hyperparameter Importance
          </h3>
          <div className="space-y-2">
            {hpImportance.importances?.slice(0, 10).map((item: any) => (
              <div key={item.parameter} className="flex justify-between text-xs">
                <span className="text-slate-400">{item.parameter}</span>
                <span className="text-neon-blue font-mono">{item.correlation.toFixed(4)}</span>
              </div>
            ))}
          </div>
        </Card>
      )}

      {metricCorr && !metricCorr.error && (
        <Card>
          <h3 className="text-sm font-display font-bold text-white mb-3">
            Metric Correlation Matrix
          </h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr>
                  <th />
                  {metricCorr.metrics.map((m: string) => (
                    <th key={m} className="px-2 text-slate-400 font-medium">{m}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metricCorr.correlation_matrix.map((row: any, i: number) => (
                  <tr key={i}>
                    <td className="text-slate-400 px-2">{metricCorr.metrics[i]}</td>
                    {row.map((val: number, j: number) => (
                      <td
                        key={j}
                        className={`px-2 text-center font-mono ${
                          val === null
                            ? 'text-slate-500'
                            : val > 0.5
                            ? 'text-green-400'
                            : val < -0.5
                            ? 'text-red-400'
                            : 'text-slate-300'
                        }`}
                      >
                        {val !== null ? val.toFixed(2) : '–'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}

      {!hpImportance && !metricCorr && (
        <p className="text-slate-500 text-sm text-center py-8">
          No analytics data available yet. Complete some experiments first.
        </p>
      )}
    </div>
  )
}