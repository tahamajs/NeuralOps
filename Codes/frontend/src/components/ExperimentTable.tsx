import { useNavigate } from "react-router-dom";
import api from "../api/client";

export default function ExperimentTable({
  experiments,
}: {
  experiments: any[];
}) {
  const nav = useNavigate();

  return (
    <div className="overflow-hidden">
      <table className="w-full text-xs">
        <thead>
          <tr className="text-left text-slate-500 font-black uppercase tracking-[0.2em] border-b border-white/5">
            <th className="px-6 py-4">Session Hash</th>
            <th className="px-6 py-4">Operation</th>
            <th className="px-6 py-4">State</th>
            <th className="px-6 py-4">Compute Time</th>
            <th className="px-6 py-4 text-right">Actions</th>
          </tr>
        </thead>
        <tbody className="divide-y divide-white/5">
          {experiments.map((exp: any) => (
            <tr
              key={exp.id}
              className="group hover:bg-white/[0.02] transition-colors"
            >
              <td className="px-6 py-5 font-mono text-blue-400">
                {exp.id.slice(0, 8)}
                <div className="text-[10px] text-slate-600 mt-1 font-sans">
                  {exp.name || "Unnamed Session"}
                </div>
              </td>
              <td className="px-6 py-5">
                <span className="px-2 py-1 bg-white/5 rounded text-white font-bold uppercase tracking-tighter">
                  {exp.job_type}
                </span>
              </td>
              <td className="px-6 py-5">
                <div className="flex items-center gap-2">
                  <div
                    className={`w-1.5 h-1.5 rounded-full ${
                      exp.status === "running"
                        ? "bg-blue-500 animate-pulse shadow-[0_0_8px_rgba(59,130,246,0.5)]"
                        : exp.status === "completed"
                          ? "bg-green-500"
                          : exp.status === "failed"
                            ? "bg-red-500"
                            : "bg-slate-500"
                    }`}
                  />
                  <span className="font-bold text-slate-300 uppercase tracking-widest text-[10px]">
                    {exp.status}
                  </span>
                </div>
              </td>
              <td className="px-6 py-5 text-slate-500 font-mono">
                {new Date(exp.created_at).toLocaleTimeString()}
              </td>
              <td className="px-6 py-5 text-right">
                <button
                  onClick={() => nav(`/experiments/${exp.id}`)}
                  className="px-4 py-2 bg-white/5 hover:bg-blue-600 hover:text-white rounded-lg font-black uppercase tracking-widest text-[10px] transition-all"
                >
                  DECODE
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
