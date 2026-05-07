import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'
import { ENDPOINTS } from '../../api/endpoints'
import { Spinner } from '../ui/Spinner'
import { File, ExternalLink } from 'lucide-react'

export const ArtifactList = ({ experimentId }: { experimentId: string }) => {
  const { data, isLoading } = useQuery({
    queryKey: ['artifacts', experimentId],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.ARTIFACTS(experimentId))).data,
  })

  if (isLoading) return <Spinner className="py-4" />
  const items = data?.artifacts || data?.files || []

  if (!items.length) {
    return (
      <div className="flex flex-col items-center py-4 text-slate-500 text-xs">
        <File size={20} className="mb-1 opacity-30" />
        No artifacts yet
      </div>
    )
  }

  return (
    <ul className="text-xs space-y-1.5">
      {Array.isArray(items)
        ? items.map((name: string) => (
            <li key={name}>
              <a
                href={`/api/experiments/${experimentId}/artifacts/${name}`}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-neon-blue hover:underline"
              >
                <File size={12} />
                {name}
                <ExternalLink size={10} className="text-slate-500" />
              </a>
            </li>
          ))
        : Object.entries(items).map(([name, url]: any) => (
            <li key={name}>
              <a
                href={url}
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-1.5 text-neon-blue hover:underline"
              >
                <File size={12} />
                {name}
                <ExternalLink size={10} className="text-slate-500" />
              </a>
            </li>
          ))}
    </ul>
  )
}