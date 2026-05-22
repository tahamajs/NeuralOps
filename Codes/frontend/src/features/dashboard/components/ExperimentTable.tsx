import { useNavigate } from 'react-router-dom'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { Experiment } from '../../../types/experiment'
import { ArrowRight, Search } from 'lucide-react'

interface ExperimentTableProps {
  experiments: Experiment[]
  showViewAll?: boolean
}

export const ExperimentTable = ({ experiments, showViewAll = true }: ExperimentTableProps) => {
  const navigate = useNavigate()

  if (experiments.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-10 text-slate-500">
        <Search size={32} className="mb-2 opacity-30" />
        <p className="text-xs">No experiments found.</p>
      </div>
    )
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-white/5">
            <th className="py-3 px-4 font-medium w-[180px]">Name</th>
            <th className="py-3 px-4 font-medium">Task</th>
            <th className="py-3 px-4 font-medium">Status</th>
            <th className="py-3 px-4 font-medium">Accuracy</th>
            <th className="py-3 px-4 font-medium">Created</th>
            {showViewAll && <th className="py-3 px-4 font-medium text-right w-[80px]">Action</th>}
           </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {experiments.map(exp => (
            <tr
              key={exp.id}
              className="hover:bg-white/5 cursor-pointer transition-colors group"
              onClick={() => navigate(`/experiments/${exp.id}`)}
            >
              <td className="py-3 px-4">
                <p className="text-white font-medium truncate max-w-[160px]">
                  {exp.name || exp.id.slice(0, 8)}
                </p>
                <p className="text-[10px] text-slate-500 font-mono">
                  {exp.id.slice(0, 8)}
                </p>
               </td>
              <td className="py-3 px-4">
                <span className="px-2 py-0.5 rounded bg-white/5 text-[10px] font-bold uppercase text-slate-300">
                  {exp.config?.task_name || '—'}
                </span>
               </td>
              <td className="py-3 px-4">
                <Badge variant={exp.status}>{exp.status}</Badge>
               </td>
              <td className="py-3 px-4 font-mono text-sm text-white">
                {exp.final_metrics?.eval_accuracy?.toFixed(4) || '—'}
               </td>
              <td className="py-3 px-4 text-slate-400 text-xs">
                {new Date(exp.created_at).toLocaleDateString()}
               </td>
              {showViewAll && (
                <td className="py-3 px-4 text-right">
                  <Button
                    variant="ghost"
                    size="sm"
                    className="opacity-0 group-hover:opacity-100 transition-opacity"
                    onClick={e => {
                      e.stopPropagation()
                      navigate(`/experiments/${exp.id}`)
                    }}
                  >
                    <ArrowRight size={14} />
                  </Button>
                 </td>
              )}
             </tr>
          ))}
        </tbody>
       </table>
    </div>
  )
}