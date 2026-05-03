import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function MethodPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useQuery({ queryKey: ['config/methods'], queryFn: async () => (await api.get('/config/methods/')).data })

  return (
    <div>
      <label className="block text-sm font-medium">Method</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full border rounded px-3 py-2">
        {(data?.methods || []).map((m: string) => (
          <option key={m} value={m}>{m}</option>
        ))}
      </select>
    </div>
  )
}