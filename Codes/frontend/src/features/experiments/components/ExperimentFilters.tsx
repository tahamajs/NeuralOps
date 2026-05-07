import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'

export interface ExperimentFiltersState {
  search: string
  status: string
  type: string
}

interface ExperimentFiltersProps {
  value: ExperimentFiltersState
  onChange: (val: ExperimentFiltersState) => void
}

export const ExperimentFilters = ({ value, onChange }: ExperimentFiltersProps) => (
  <div className="flex flex-wrap gap-3 glass p-4">
    <Input
      placeholder="Search by name or ID"
      className="flex-1 min-w-[200px]"
      value={value.search}
      onChange={(e) => onChange({ ...value, search: e.target.value })}
    />
    <Select
      options={[
        { value: '', label: 'All Status' },
        { value: 'queued', label: 'Queued' },
        { value: 'running', label: 'Running' },
        { value: 'completed', label: 'Completed' },
        { value: 'failed', label: 'Failed' },
        { value: 'stopped', label: 'Stopped' },
      ]}
      value={value.status}
      onChange={(e) => onChange({ ...value, status: e.target.value })}
      className="w-40"
    />
    <Select
      options={[
        { value: '', label: 'All Types' },
        { value: 'training', label: 'Training' },
        { value: 'evaluation', label: 'Evaluation' },
        { value: 'compression', label: 'Compression' },
      ]}
      value={value.type}
      onChange={(e) => onChange({ ...value, type: e.target.value })}
      className="w-40"
    />
  </div>
)