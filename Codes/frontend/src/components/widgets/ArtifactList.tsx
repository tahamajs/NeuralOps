import { useQuery } from '@tanstack/react-query'
import api from '../../api/client'
import { ENDPOINTS } from '../../api/endpoints'
import { Spinner } from '../ui/Spinner'

export const ArtifactList = ({ experimentId }: { experimentId: string }) => {
  const { data, isLoading } = useQuery({
    queryKey: ['artifacts', experimentId],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.ARTIFACTS(experimentId))).data,
  })

  if (isLoading) return <Spinner className="py-4" />
  const items = data?.artifacts || data?.files || []
  if (!items.length) return <p className="text-xs text-slate-500">No artifacts</p>

  return (
    <ul className="text-xs space-y-1">
      {Object.entries(items).map(([name, url]: [string, string]) => (
        <li key={name}>
          <a
            href={url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-neon-blue hover:underline"
          >
            {name}
          </a>
        </li>
      ))}
    </ul>
  )
}