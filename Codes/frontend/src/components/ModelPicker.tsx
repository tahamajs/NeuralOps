import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function ModelPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useQuery({ queryKey: ['config/models'], queryFn: async () => (await api.get('/config/models/')).data })

  return (
    <div>
      <label className="block text-sm font-medium">Model</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full border rounded px-3 py-2">
        {(data?.models || []).map((m: string) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </div>
  )
}