import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'
import { LiveLossPlot } from '../../../components/widgets/LiveLossPlot'

export default function JobHistoryPage() {
  const { jobId } = useParams<{ jobId: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['jobHistory', jobId],
    queryFn: async () => (await api.get(`/api/jobs/${jobId}/history/`)).data,
  })

  if (isLoading) return <Spinner />
  const steps = data?.steps || []
  const losses = data?.losses || []
  const evalSteps = data?.eval_steps || []
  const evalLosses = data?.eval_losses || []
  const evalAccuracies = data?.eval_accuracies || []

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        TRAINING <span className="text-neon-blue">HISTORY</span>
      </h1>
      {steps.length === 0 ? (
        <p className="text-slate-500">No training data recorded yet.</p>
      ) : (
        <Card>
          <LiveLossPlot metrics={{ loss: losses, accuracy: evalAccuracies }} />
          {evalSteps.length > 0 && (
            <div className="mt-4">
              <h3 className="text-sm font-bold text-white">Evaluation History</h3>
              <div className="overflow-x-auto">
                <table className="w-full text-xs mt-2">
                  <thead>
                    <tr className="text-left text-slate-400">
                      <th className="py-1 px-2">Step</th>
                      <th className="py-1 px-2">Loss</th>
                      <th className="py-1 px-2">Accuracy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {evalSteps.map((step: number, i: number) => (
                      <tr key={step} className="border-b border-white/5">
                        <td className="py-1 px-2">{step}</td>
                        <td className="py-1 px-2">{evalLosses[i]?.toFixed(4)}</td>
                        <td className="py-1 px-2">{evalAccuracies[i]?.toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </Card>
      )}
    </div>
  )
}