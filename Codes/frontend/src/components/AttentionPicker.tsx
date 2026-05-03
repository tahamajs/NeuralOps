import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function AttentionPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useQuery({ queryKey: ['config/attentions'], queryFn: async () => (await api.get('/config/attentions/')).data })

  return (
    <div>
      <label className="block text-sm font-medium">Attention Type</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full border rounded px-3 py-2">
        {(data?.attentions || []).map((a: string) => (
          <option key={a} value={a}>{a}</option>
        ))}
      </select>
    </div>
  )
}