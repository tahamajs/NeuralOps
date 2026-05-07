import { useNavigate } from 'react-router-dom'
import { Badge } from '../../../components/ui/Badge'
import { Experiment } from '../../../types/experiment'

export const ExperimentTable = ({ experiments }: { experiments: Experiment[] }) => {
  const nav = useNavigate()
  if (!experiments.length) return <p className="text-slate-500 text-xs text-center py-4">No experiments found.</p>
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-400 uppercase tracking-wider border-b border-white/5">
            <th className="py-3 px-4">Name</th>
            <th className="py-3 px-4">Task</th>
            <th className="py-3 px-4">Status</th>
            <th className="py-3 px-4">Accuracy</th>
            <th className="py-3 px-4">Created</th>
            <th className="py-3 px-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {experiments.map(exp => (
            <tr key={exp.id} className="hover:bg-white/5 cursor-pointer" onClick={() => nav(`/experiments/${exp.id}`)}>
              <td className="py-3 px-4 text-white">{exp.name || exp.id.slice(0,8)}</td>
              <td className="py-3 px-4 uppercase">{exp.config?.task_name || '—'}</td>
              <td className="py-3 px-4"><Badge status={exp.status}>{exp.status}</Badge></td>
              <td className="py-3 px-4">{exp.final_metrics?.eval_accuracy?.toFixed(4) || '—'}</td>
              <td className="py-3 px-4 text-slate-400">{new Date(exp.created_at).toLocaleDateString()}</td>
              <td className="py-3 px-4 text-right"><button className="text-neon-blue hover:text-white text-[10px] font-bold uppercase tracking-wider">View →</button></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}