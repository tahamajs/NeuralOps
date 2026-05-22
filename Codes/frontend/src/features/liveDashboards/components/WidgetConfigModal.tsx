import { useState } from 'react'
import { Modal } from '../../../components/ui/Modal'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Button } from '../../../components/ui/Button'
import { Widget } from '../types'

export function WidgetConfigModal({ open, widget, onClose, onSave }: any) {
  const [title, setTitle] = useState(widget?.title || '')
  const [dataSourceType, setDataSourceType] = useState(widget?.data_source.type || 'experiment')
  const [dataSourceId, setDataSourceId] = useState(widget?.data_source.id || '')
  const [metricKey, setMetricKey] = useState(widget?.data_source.metric_key || '')

  const handleSave = () => {
    onSave({
      ...widget,
      title,
      data_source: { type: dataSourceType, id: dataSourceId || undefined, metric_key: metricKey || undefined },
    })
  }

  return (
    <Modal open={open} onClose={onClose} title="Configure Widget">
      <div className="space-y-4">
        <Input label="Title" value={title} onChange={e => setTitle(e.target.value)} />
        <Select label="Data Source" options={[{ value: 'experiment', label: 'Experiment' }, { value: 'run', label: 'Run' }, { value: 'system', label: 'System Metrics' }]} value={dataSourceType} onChange={e => setDataSourceType(e.target.value)} />
        {dataSourceType !== 'system' && <Input label="ID" value={dataSourceId} onChange={e => setDataSourceId(e.target.value)} placeholder="Experiment or Run ID" />}
        {widget?.type === 'metric' && <Input label="Metric Key" value={metricKey} onChange={e => setMetricKey(e.target.value)} placeholder="e.g., accuracy, loss" />}
        <Button onClick={handleSave}>Save</Button>
      </div>
    </Modal>
  )
}