import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Select } from '../../../components/ui/Select'

export const ModelPicker = ({ value, onChange }: { value: string; onChange: (v: string) => void }) => {
  const { data } = useQuery({ queryKey: ['config/models'], queryFn: async () => (await api.get(ENDPOINTS.CONFIG.MODELS)).data })
  const models = data?.models || []
  return (
    <Select
      label="Foundation Model"
      options={models.map((m: string) => ({ value: m, label: m }))}
      value={value}
      onChange={(e) => onChange(e.target.value)}
    />
  )
}