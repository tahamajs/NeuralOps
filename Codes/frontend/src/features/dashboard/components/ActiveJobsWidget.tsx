import { useNavigate } from 'react-router-dom'
import { Card } from '../../../components/ui/Card'
import { Experiment } from '../../../types/experiment'

export const ActiveJobsWidget = ({ experiments }: { experiments: Experiment[] }) => {
  const nav = useNavigate()
  const active = experiments.filter((e) => e.status === 'running' || e.status === 'queued')

  return (
    <Card>
      <h3 className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-3">Active Jobs</h3>
      {active.length === 0 && <div className="text-xs text-slate-600">No active jobs</div>}
      {active.map((job) => (
        <div
          key={job.id}
          className="flex items-center justify-between py-1.5 cursor-pointer hover:bg-white/5 rounded px-2 -mx-2 transition-colors"
          onClick={() => nav(`/experiments/${job.id}`)}
        >
          <span className="text-sm font-medium text-white truncate">{job.name || job.id.slice(0, 8)}</span>
          <span className="text-[10px] uppercase tracking-wider ml-2 text-slate-400">{job.status}</span>
        </div>
      ))}
    </Card>
  )
}