import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Spinner, Table } from '../../../components/ui'
import { Trophy } from 'lucide-react'

export default function LeaderboardPage() {
  const { data, isLoading } = useQuery({
    queryKey: ['leaderboard'],
    queryFn: async () => (await api.get('/leaderboard/')).data,
  })
  if (isLoading) return <Spinner />
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white flex items-center gap-2"><Trophy size={24} className="text-yellow-400" /> Global Leaderboard</h1>
      <Card>
        <Table>
          <thead><tr><th>Rank</th><th>User</th><th>Score</th><th>Contributions</th></tr></thead>
          <tbody>
            {data?.map((entry: any, idx: number) => (
              <tr key={entry.user_id}>
                <td>{idx+1}</td>
                <td className="font-bold">{entry.username}</td>
                <td className="text-neon-blue">{entry.total_score}</td>
                <td>{entry.job_count}</td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  )
}