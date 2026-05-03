import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function ArtifactList({ experimentId }: { experimentId: string }) {
  const { data } = useQuery({
    queryKey: ['artifacts', experimentId],
    queryFn: async () => (await api.get(`/experiments/${experimentId}/artifacts/`)).data,
  })

  if (!data?.artifacts?.length) return null

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">Artifacts</h2>
      <ul className="text-sm space-y-1">
        {data.artifacts.map((a: string) => (
          <li key={a}>
            <a href={`/api/experiments/${experimentId}/artifacts/${a}`} download className="text-blue-600 hover:underline">
              {a}
            </a>
          </li>
        ))}
      </ul>
    </div>
  )
}