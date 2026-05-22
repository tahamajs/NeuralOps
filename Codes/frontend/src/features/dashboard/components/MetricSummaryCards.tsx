import { Card } from '../../../components/ui/Card'
import { Experiment } from '../../../types/experiment'

interface MetricSummaryCardsProps {
  experiments: Experiment[]
}

const METRICS = [
  { key: 'queued', label: 'Queued', color: 'text-yellow-400' },
  { key: 'running', label: 'Running', color: 'text-blue-400' },
  { key: 'completed', label: 'Completed', color: 'text-green-400' },
  { key: 'failed', label: 'Failed', color: 'text-red-400' },
] as const

export const MetricSummaryCards = ({ experiments }: MetricSummaryCardsProps) => {
  const counts = {
    queued: experiments.filter(e => e.status === 'queued').length,
    running: experiments.filter(e => e.status === 'running').length,
    completed: experiments.filter(e => e.status === 'completed').length,
    failed: experiments.filter(e => e.status === 'failed').length,
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {METRICS.map(({ key, label, color }) => (
        <Card key={key} className="p-5 text-center card-hover-effect">
          <div className={`text-3xl font-display font-bold ${color}`}>
            {counts[key]}
          </div>
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">
            {label}
          </div>
        </Card>
      ))}
    </div>
  )
}