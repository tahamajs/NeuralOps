import { useQueries } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

const configEndpoints = [
  { key: 'attentions', path: '/api/config/attentions/' },
  { key: 'compressors', path: '/api/config/compressors/' },
  { key: 'trainers', path: '/api/config/trainers/' },
  { key: 'methods', path: '/api/config/methods/' },
  { key: 'models', path: '/api/config/models/' },
  { key: 'tasks', path: '/api/config/tasks/' },
  { key: 'losses', path: '/api/config/losses/' },
  { key: 'optimizers', path: '/api/config/optimizers/' },
  { key: 'schedulers', path: '/api/config/schedulers/' },
  { key: 'callbacks', path: '/api/config/callbacks/' },
  { key: 'augmentations', path: '/api/config/augmentations/' },
]

export default function ConfigPage() {
  const results = useQueries({
    queries: configEndpoints.map(({ key, path }) => ({
      queryKey: ['config', key],
      queryFn: async () => (await api.get(path)).data,
    })),
  })

  const isLoading = results.some(r => r.isLoading)
  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        CONFIGURATION <span className="text-neon-blue">REFERENCE</span>
      </h1>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {configEndpoints.map((ep, idx) => {
          const data = results[idx].data
          const values = data ? (data[ep.key] || data.models || data.tasks || []) : []
          return (
            <Card key={ep.key}>
              <h3 className="text-sm font-bold text-white capitalize mb-2">{ep.key}</h3>
              <ul className="text-xs text-slate-400 max-h-32 overflow-y-auto scrollbar-thin">
                {Array.isArray(values) ? (
                  values.map((v: string, i: number) => <li key={i}>• {v}</li>)
                ) : (
                  Object.entries(values).map(([k, v], i) => <li key={i}>• {k}: {typeof v === 'string' ? v : JSON.stringify(v)}</li>)
                )}
              </ul>
            </Card>
          )
        })}
      </div>
    </div>
  )
}