import { useNavigate } from 'react-router-dom'
import StatusBadge from './StatusBadge'
import api from '../api/client'

export default function ExperimentTable({ experiments }: { experiments: any[] }) {
  const nav = useNavigate()

  return (
    <div className="bg-white rounded-lg shadow overflow-x-auto">
      <table className="w-full text-sm">
        <thead className="text-left text-gray-500 bg-gray-50">
          <tr>
            <th className="px-4 py-2">Name</th>
            <th className="px-4 py-2">Type</th>
            <th className="px-4 py-2">Status</th>
            <th className="px-4 py-2">Created</th>
            <th className="px-4 py-2">Actions</th>
          </tr>
        </thead>
        <tbody>
          {experiments.map((exp: any) => (
            <tr key={exp.id} className="border-t hover:bg-gray-50">
              <td className="px-4 py-2">{exp.name || exp.id.slice(0, 8)}</td>
              <td className="px-4 py-2 capitalize">{exp.job_type || 'training'}</td>
              <td className="px-4 py-2"><StatusBadge status={exp.status} /></td>
              <td className="px-4 py-2">{new Date(exp.created_at).toLocaleDateString()}</td>
              <td className="px-4 py-2">
                <button onClick={() => nav(`/experiments/${exp.id}`)} className="text-blue-600 hover:underline">
                  View
                </button>
                {exp.status === 'running' && (
                  <button
                    onClick={() => api.post(`/experiments/${exp.id}/stop/`)}
                    className="text-red-600 hover:underline ml-2"
                  >
                    Stop
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}