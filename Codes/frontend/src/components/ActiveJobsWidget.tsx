import { useNavigate } from 'react-router-dom'

export default function ActiveJobsWidget({ experiments }: { experiments: any[] }) {
  const nav = useNavigate()
  const active = experiments.filter((e) => e.status === 'running' || e.status === 'queued')

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Active Jobs</h2>
      {active.length === 0 && <div className="text-sm text-gray-500">No active jobs</div>}
      {active.map((job: any) => (
        <div
          key={job.id}
          className="text-sm py-1 cursor-pointer hover:bg-gray-50"
          onClick={() => nav(`/experiments/${job.id}`)}
        >
          <span className="font-medium">{job.name || job.id.slice(0, 8)}</span>
          <span className="text-gray-500 ml-2 capitalize">({job.status})</span>
        </div>
      ))}
    </div>
  )
}