export default function ConfigPreviewPanel({ config }: { config: any }) {
  return (
    <div className="bg-gray-50 rounded-lg shadow p-4">
      <h2 className="text-sm font-medium mb-1">Config Preview</h2>
      <pre className="text-xs text-gray-600 overflow-auto max-h-32">{JSON.stringify(config, null, 2)}</pre>
    </div>
  )
}