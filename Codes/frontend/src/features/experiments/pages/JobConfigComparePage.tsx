import { useParams } from 'react-router-dom'
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'

export default function JobConfigComparePage() {
  const { jobId } = useParams<{ jobId: string }>()
  const [va, setVa] = useState('')
  const [vb, setVb] = useState('')
  const [diff, setDiff] = useState<any>(null)

  const { data: history } = useQuery({
    queryKey: ['configHistory', jobId],
    queryFn: async () => (await api.get(`/api/jobs/${jobId}/config/history/`)).data,
  })
  const versions = history?.versions || []

  const handleCompare = async () => {
    if (!va || !vb) return
    const res = await api.get(`/api/jobs/${jobId}/config/compare/${va}/${vb}/`)
    setDiff(res.data)
  }

  const options = versions.map((v: any) => ({ value: v.version.toString(), label: `v${v.version}` }))

  return (
    <div className="space-y-6 animate-fade-in max-w-4xl mx-auto">
      <h1 className="text-2xl font-black text-white">
        CONFIG <span className="text-neon-blue">DIFF</span>
      </h1>
      <Card>
        <div className="flex gap-4 mb-4">
          <Select label="Version A" options={options} value={va} onChange={(e) => setVa(e.target.value)} />
          <Select label="Version B" options={options} value={vb} onChange={(e) => setVb(e.target.value)} />
          <Button onClick={handleCompare} className="self-end">Compare</Button>
        </div>
        {diff && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div>
              <h4 className="text-white font-bold mb-2">Values Changed</h4>
              <pre className="bg-cyber-800 p-2 rounded text-slate-400">{JSON.stringify(diff.values_changed, null, 2)}</pre>
            </div>
            <div>
              <h4 className="text-white font-bold mb-2">Added</h4>
              <pre className="bg-cyber-800 p-2 rounded text-slate-400">{JSON.stringify(diff.dictionary_item_added, null, 2)}</pre>
            </div>
            <div>
              <h4 className="text-white font-bold mb-2">Removed</h4>
              <pre className="bg-cyber-800 p-2 rounded text-slate-400">{JSON.stringify(diff.dictionary_item_removed, null, 2)}</pre>
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}