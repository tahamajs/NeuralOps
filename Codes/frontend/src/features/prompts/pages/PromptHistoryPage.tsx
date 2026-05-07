import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function PromptHistoryPage() {
  const { name } = useParams<{ name: string }>()
  const { data, isLoading } = useQuery({
    queryKey: ['promptHistory', name],
    queryFn: async () => (await api.get(`/api/prompts/${name}/history/`)).data,
  })

  if (isLoading) return <Spinner />
  const history = data || []

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        PROMPT <span className="text-neon-blue">HISTORY</span>: {name}
      </h1>
      {history.length === 0 ? (
        <p className="text-slate-500">No history available.</p>
      ) : (
        history.map((item: any, i: number) => (
          <Card key={i} className="flex justify-between items-center">
            <div>
              <span className="text-neon-blue font-mono">v{item.version}</span>
              <p className="text-xs text-slate-400 mt-1">{item.template?.slice(0, 100)}</p>
            </div>
            <span className="text-xs text-slate-500">{new Date(item.timestamp).toLocaleString()}</span>
          </Card>
        ))
      )}
    </div>
  )
}