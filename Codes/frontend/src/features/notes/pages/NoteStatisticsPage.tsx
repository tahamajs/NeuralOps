import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Spinner } from '../../../components/ui/Spinner'

export default function NoteStatisticsPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['noteStats'],
    queryFn: async () => (await api.get('/api/notes/v1/notes/statistics/')).data,
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">
        NOTES <span className="text-neon-blue">STATISTICS</span>
      </h1>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Total Notes</div>
          <div className="text-2xl font-bold text-white">{data.total_notes}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Public</div>
          <div className="text-2xl font-bold text-green-400">{data.public_notes}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Private</div>
          <div className="text-2xl font-bold text-slate-400">{data.private_notes}</div>
        </Card>
        <Card>
          <div className="text-xs text-slate-500 uppercase tracking-widest">Top Authors</div>
          <div className="text-sm text-white mt-1 space-y-1">
            {data.top_authors?.slice(0,3).map((a: any) => (
              <div key={a.author__username}>{a.author__username}: {a.count}</div>
            ))}
          </div>
        </Card>
      </div>
      {data.top_runs && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-2">Most Active Runs</h3>
          {data.top_runs.map((r: any) => (
            <div key={r.run__id} className="flex justify-between text-xs py-1 border-b border-white/5">
              <span className="font-mono text-neon-blue">{r.run__id.slice(0,8)}</span>
              <span className="font-bold">{r.count} notes</span>
            </div>
          ))}
        </Card>
      )}
    </div>
  )
}