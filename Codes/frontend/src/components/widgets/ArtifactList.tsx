import { File } from 'lucide-react'

interface Artifact {
  name: string
  size?: number
  url?: string
}

export function ArtifactList({ artifacts }: { artifacts?: Artifact[] }) {
  if (!artifacts || artifacts.length === 0) {
    return <div className="text-sm text-slate-500">No artifacts found.</div>
  }

  return (
    <div className="space-y-2">
      <h3 className="text-xs font-bold text-slate-400 uppercase tracking-wider mb-2">
        Artifacts
      </h3>
      {artifacts.map((a, i) => (
        <a
          key={i}
          href={a.url || '#'}
          target="_blank"
          rel="noopener noreferrer"
          className="flex items-center gap-2 text-sm text-neon-blue hover:underline"
        >
          <File size={14} />
          {a.name}
          {a.size && <span className="text-slate-500 text-xs ml-auto">{a.size} MB</span>}
        </a>
      ))}
    </div>
  )
}