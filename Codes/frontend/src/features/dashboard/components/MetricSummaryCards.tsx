import { Card } from '../../../components/ui/Card'
import { Experiment } from '../../../types/experiment'

export const MetricSummaryCards = ({ experiments }: { experiments: Experiment[] }) => {
  const counts = {
    queued: experiments.filter((e) => e.status === 'queued').length,
    running: experiments.filter((e) => e.status === 'running').length,
    completed: experiments.filter((e) => e.status === 'completed').length,
    failed: experiments.filter((e) => e.status === 'failed').length,
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      {Object.entries(counts).map(([key, val]) => (
        <Card key={key} className="p-5 text-center" hover={false}>
          <div className="text-2xl font-black text-white">{val}</div>
          <div className="text-[10px] font-bold text-slate-500 uppercase tracking-widest mt-1">{key}</div>
        </Card>
      ))}
    </div>
  )
}