export default function ExperimentFilters({
  value,
  onChange,
}: {
  value: any
  onChange: (val: any) => void
}) {
  return (
    <div className="flex gap-3 bg-white p-3 rounded shadow">
      <input
        type="text"
        placeholder="Search by name"
        className="border rounded px-3 py-2 flex-1"
        value={value.search}
        onChange={(e) => onChange({ ...value, search: e.target.value })}
      />
      <select
        className="border rounded px-3 py-2"
        value={value.status}
        onChange={(e) => onChange({ ...value, status: e.target.value })}
      >
        <option value="">All Status</option>
        <option value="queued">Queued</option>
        <option value="running">Running</option>
        <option value="completed">Completed</option>
        <option value="failed">Failed</option>
        <option value="stopped">Stopped</option>
      </select>
      <select
        className="border rounded px-3 py-2"
        value={value.type}
        onChange={(e) => onChange({ ...value, type: e.target.value })}
      >
        <option value="">All Types</option>
        <option value="training">Training</option>
        <option value="evaluation">Evaluation</option>
        <option value="compression">Compression</option>
      </select>
    </div>
  )
}