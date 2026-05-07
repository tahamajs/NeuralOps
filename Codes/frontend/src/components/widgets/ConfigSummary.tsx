export const ConfigSummary = ({ config }: { config: Record<string, any> }) => (
  <pre className="text-xs text-slate-400 overflow-auto max-h-64 bg-white/5 rounded-lg p-3 scrollbar-thin">
    {JSON.stringify(config, null, 2)}
  </pre>
)