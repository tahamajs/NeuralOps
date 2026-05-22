import { formatDistanceToNow } from 'date-fns'

export function VersionTimeline({ versions }: any) {
  return <div className="space-y-4">{versions.map(v => (<div key={v.version} className="border-l-2 border-neon-blue pl-4"><div className="text-xs text-slate-400">v{v.version} · {formatDistanceToNow(new Date(v.created_at), { addSuffix: true })}</div><div className="font-medium text-white">{v.author}</div><ul className="list-disc list-inside text-sm text-slate-300">{v.changes.map((c: string, i: number) => <li key={i}>{c}</li>)}</ul></div>))}</div>
}