import { Code2 } from 'lucide-react'

export const ConfigSummary = ({ config }: { config: Record<string, any> }) => (
  <div>
    {!config || Object.keys(config).length === 0 ? (
      <div className="flex items-center gap-2 text-slate-500 text-xs py-2">
        <Code2 size={14} />
        No configuration available
      </div>
    ) : (
      <pre className="text-xs text-slate-400 overflow-auto max-h-64 bg-white/5 rounded-lg p-3 scrollbar-thin">
        {JSON.stringify(config, null, 2)}
      </pre>
    )}
  </div>
)