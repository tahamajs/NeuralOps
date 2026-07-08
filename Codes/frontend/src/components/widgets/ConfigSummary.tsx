export function ConfigSummary({ config }: { config?: Record<string, any> }) {
  if (!config)
    return (
      <div className="text-sm text-slate-500">No configuration available.</div>
    );

  const entries = Object.entries(config).slice(0, 10);
  return (
    <div className="space-y-1">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Configuration
      </h3>
      {entries.map(([key, value]) => (
        <div key={key} className="flex justify-between text-xs">
          <span className="text-slate-500">{key}</span>
          <span className="text-white font-mono">{String(value)}</span>
        </div>
      ))}
    </div>
  );
}
