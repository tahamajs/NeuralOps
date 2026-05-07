import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Select } from '../../../components/ui/Select'

export const MethodPicker = ({ value, onChange }: { value: string; onChange: (v: string) => void }) => {
  const { data } = useQuery({ queryKey: ['config/methods'], queryFn: async () => (await api.get(ENDPOINTS.CONFIG.METHODS)).data })
  const methods: string[] = data?.methods || []
  return (
    <Select
      label="Training Method"
      options={methods.map((m) => ({ value: m, label: m }))}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}