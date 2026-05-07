import { useNavigate } from 'react-router-dom'
import { Badge } from '../../../components/ui/Badge'
import { Experiment } from '../../../types/experiment'

export const ExperimentTable = ({ experiments }: { experiments: Experiment[] }) => {
  const nav = useNavigate()
  if (experiments.length === 0) return <div className="py-8 text-center text-slate-500 text-xs">No experiments yet</div>

  return (
    <div className="overflow-x-auto scrollbar-thin">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500 font-black uppercase tracking-[0.15em] border-b border-white/5">
            <th className="px-6 py-3">Session</th>
            <th className="px-6 py-3">Type</th>
            <th className="px-6 py-3">Status</th>
            <th className="px-6 py-3">Started</th>
            <th className="px-6 py-3 text-right">Action</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {experiments.map((exp) => (
            <tr
              key={exp.id}
              className="group hover:bg-white/5 transition-colors cursor-pointer"
              onClick={() => nav(`/experiments/${exp.id}`)}
            >
              <td className="px-6 py-4">
                <div className="font-mono text-neon-blue text-xs">{exp.id.slice(0, 8)}</div>
                <div className="text-slate-500 text-[10px] mt-0.5 font-medium">{exp.name || 'Unnamed session'}</div>
              </td>
              <td className="px-6 py-4">
                <span className="px-2 py-0.5 bg-white/10 rounded text-white font-bold uppercase text-[10px]">
                  {exp.job_type}
                </span>
              </td>
              <td className="px-6 py-4">
                <Badge status={exp.status} />
              </td>
              <td className="px-6 py-4 text-slate-400 font-mono text-[10px]">
                {new Date(exp.created_at).toLocaleTimeString()}
              </td>
              <td className="px-6 py-4 text-right">
                <button className="text-neon-blue hover:text-white text-[10px] font-bold uppercase tracking-widest transition-colors">
                  Decode →
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}