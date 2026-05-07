import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'
import { Input } from '../../../components/ui/Input' // not used but we'll use textarea

export default function DriftCheckPage() {
  const { runId } = useParams<{ runId: string }>()
  const [reference, setReference] = useState('{}')
  const [current, setCurrent] = useState('{}')
  const [result, setResult] = useState<any>(null)

  const mutation = useMutation({
    mutationFn: async () => {
      const res = await api.post(`/api/runs/${runId}/drift/check/`, {
        reference: JSON.parse(reference),
        current: JSON.parse(current),
      })
      return res.data
    },
    onSuccess: (data) => setResult(data),
  })

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white tracking-tighter">
        DRIFT <span className="text-neon-blue">DETECTION</span>
      </h1>
      <Card>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">
              Reference Data (JSON)
            </label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white font-mono text-xs h-32 focus:border-neon-blue/50"
              value={reference}
              onChange={(e) => setReference(e.target.value)}
            />
          </div>
          <div>
            <label className="text-[10px] font-bold text-slate-500 uppercase tracking-widest block mb-1">
              Current Data (JSON)
            </label>
            <textarea
              className="w-full bg-white/5 border border-white/10 rounded-xl p-3 text-white font-mono text-xs h-32 focus:border-neon-blue/50"
              value={current}
              onChange={(e) => setCurrent(e.target.value)}
            />
          </div>
        </div>
        <Button
          onClick={() => mutation.mutate()}
          disabled={mutation.isPending}
          className="w-full"
        >
          {mutation.isPending ? 'Running...' : 'Run Drift Check'}
        </Button>
        {result && (
          <div className="mt-6">
            <h3 className="text-sm font-bold text-white mb-2">Result</h3>
            <pre className="bg-cyber-800 p-4 rounded-lg text-xs text-slate-400 overflow-auto max-h-96">
              {JSON.stringify(result, null, 2)}
            </pre>
          </div>
        )}
      </Card>
    </div>
  )
}