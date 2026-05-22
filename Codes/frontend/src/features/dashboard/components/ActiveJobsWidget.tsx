import { useNavigate } from 'react-router-dom'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Box, ArrowRight } from 'lucide-react'
import { Experiment } from '../../../types/experiment'

interface ActiveJobsWidgetProps {
  experiments: Experiment[]
}

export const ActiveJobsWidget = ({ experiments }: ActiveJobsWidgetProps) => {
  const navigate = useNavigate()
  const active = experiments.filter(e => e.status === 'running' || e.status === 'queued')

  return (
    <Card className="h-full">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-display font-bold text-white flex items-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-orange-400 animate-pulse" />
          Active Jobs
        </h3>
        <span className="text-[10px] text-slate-400">
          {active.length} job{active.length !== 1 ? 's' : ''}
        </span>
      </div>

      {active.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-6 text-slate-500">
          <Box size={32} className="mb-2 opacity-30" />
          <p className="text-xs">No active jobs</p>
        </div>
      ) : (
        <div className="space-y-2">
          {active.map(job => (
            <div
              key={job.id}
              className="flex items-center justify-between py-2 px-2 -mx-2 rounded-lg cursor-pointer hover:bg-white/5 transition-colors group"
              onClick={() => navigate(`/experiments/${job.id}`)}
            >
              <div className="flex-1 min-w-0">
                <p className="text-sm font-medium text-white truncate">
                  {job.name || job.id.slice(0, 8)}
                </p>
                <p className="text-[10px] text-slate-400 mt-0.5">
                  {job.config?.task_name?.toUpperCase() || '—'}
                </p>
              </div>
              <Badge variant={job.status}>{job.status}</Badge>
              <Button
                variant="ghost"
                size="sm"
                className="ml-2 opacity-0 group-hover:opacity-100 transition-opacity"
                onClick={e => {
                  e.stopPropagation()
                  navigate(`/experiments/${job.id}`)
                }}
              >
                <ArrowRight size={14} />
              </Button>
            </div>
          ))}
        </div>
      )}
    </Card>
  )
}