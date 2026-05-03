export default function ConfigSummary({ config }: { config: any }) {
  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Configuration</h2>
      <pre className="text-xs text-gray-700 overflow-auto max-h-64">{JSON.stringify(config, null, 2)}</pre>
    </div>
  )
}