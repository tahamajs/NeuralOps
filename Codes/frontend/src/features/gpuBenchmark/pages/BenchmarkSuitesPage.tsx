import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Spinner, Table, Badge } from '../../../components/ui'
import { useState } from 'react'

export default function BenchmarkSuitesPage() {
  const [selectedListingId, setSelectedListingId] = useState('')
  const { data: suites, isLoading } = useQuery({
    queryKey: ['benchmark-suites'],
    queryFn: async () => (await api.get('/suites/')).data,
  })
  const { data: listings } = useQuery({
    queryKey: ['marketplace-listings'],
    queryFn: async () => (await api.get('/listings/')).data,
  })
  const { data: scores } = useQuery({
    queryKey: ['benchmark-scores'],
    queryFn: async () => (await api.get('/scores/')).data,
  })

  const runBenchmark = useMutation({
    mutationFn: (listingId: string) => api.post(`/listings/${listingId}/run/`),
    onSuccess: () => alert('Benchmark started'),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold text-white">GPU Benchmarking</h1>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Benchmark Suites</h2>
        <div className="grid md:grid-cols-3 gap-3">
          {suites?.map((suite: any) => (
            <div key={suite.id} className="glass p-3 rounded">
              <h3 className="font-bold text-white">{suite.name}</h3>
              <p className="text-xs text-slate-400">{suite.description}</p>
              <p className="text-xs mt-1">Tests: {suite.test_count}</p>
              <Button size="sm" className="mt-2" onClick={() => runBenchmark.mutate(selectedListingId)} disabled={!selectedListingId}>Run on selected GPU</Button>
            </div>
          ))}
        </div>
      </Card>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Available GPU Listings</h2>
        <select className="w-full bg-white/5 border rounded p-2 mb-3" onChange={e => setSelectedListingId(e.target.value)}>
          <option value="">Select a GPU listing...</option>
          {listings?.map((l: any) => <option key={l.id} value={l.id}>{l.gpu_model} – {l.price_per_hour} tokens/h</option>)}
        </select>
      </Card>

      <Card>
        <h2 className="text-sm font-bold text-white mb-3">Leaderboard</h2>
        <Table>
          <thead><tr><th>Listing</th><th>Suite</th><th>Score</th><th>Verified</th></tr></thead>
          <tbody>
            {scores?.map((score: any) => (
              <tr key={score.id}>
                <td>{score.listing_gpu_model}</td>
                <td>{score.suite_name}</td>
                <td className="text-neon-blue font-bold">{score.score}</td>
                <td><Badge variant={score.verified ? 'success' : 'warning'}>{score.verified ? 'Verified' : 'Pending'}</Badge></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  )
}