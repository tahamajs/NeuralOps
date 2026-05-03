import { useQuery } from '@tanstack/react-query'
import api from '../api/client'

export default function TaskPicker({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const { data } = useQuery({ queryKey: ['config/tasks'], queryFn: async () => (await api.get('/config/tasks/')).data })

  return (
    <div>
      <label className="block text-sm font-medium">Task</label>
      <select value={value} onChange={(e) => onChange(e.target.value)} className="w-full border rounded px-3 py-2">
        {(data?.tasks || []).map((t: string) => (
          <option key={t} value={t}>{t}</option>
        ))}
      </select>
    </div>
  )
}