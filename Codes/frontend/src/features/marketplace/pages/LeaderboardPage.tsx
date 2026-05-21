import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'

export default function LeaderboardPage() {
  const { data } = useQuery({
    queryKey: ['marketplace-leaderboard'],
    queryFn: () => api.get('/marketplace/leaderboard/').then(r => r.data),
  })

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">GPU Leaderboard</h1>
      <div className="space-y-2">
        {data?.map((entry: any) => (
          <div key={entry.rank} className="glass p-4 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <span className="text-2xl font-bold text-slate-500">#{entry.rank}</span>
              <div>
                <p className="font-bold">{entry.gpu_model}</p>
                <p className="text-sm text-slate-400">by {entry.provider} · ${entry.price_per_hour}/hr</p>
              </div>
            </div>
            <div className="text-right">
              <p className="font-bold text-lg">{entry.composite_score}</p>
              <p className="text-xs text-slate-500">Training: {entry.training_score} · Inference: {entry.inference_score}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}