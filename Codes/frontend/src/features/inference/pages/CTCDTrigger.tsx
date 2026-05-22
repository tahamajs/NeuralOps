// src/features/inference/pages/CTCDTrigger.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { inferenceAPI } from '../services/api'
import { CTCDCard } from '../components/CTCDCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Repeat, Plus, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function CTCDTrigger({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [formData, setFormData] = useState({ name: '', type: 'cron', config: { cron_expression: '0 * * * *' }, pipeline_id: 'train', enabled: true })
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: triggers, isLoading, refetch } = useQuery({
    queryKey: ['ctc-triggers'],
    queryFn: () => inferenceAPI.getCTCTriggers(),
    refetchInterval: 30000,
  })

  const createTrigger = useMutation({
    mutationFn: () => inferenceAPI.createCTCTrigger(formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['ctc-triggers'] })
      setShowCreateModal(false)
      success('Trigger created')
    },
  })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Repeat size={20} className="text-neon-mint" /> CTC / CD Triggers</h1><p className="text-xs text-slate-400">Continuous Training & Deployment</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => refetch()}><RefreshCw size={14} /></Button><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Trigger</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid md:grid-cols-2 gap-4">{triggers?.map(t => <CTCDCard key={t.id} trigger={t} onRefresh={() => queryClient.invalidateQueries({ queryKey: ['ctc-triggers'] })} />)}</div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create CTC Trigger">
        <div className="space-y-4"><Input label="Name" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} /><Select label="Type" options={[{ value: 'cron', label: 'Cron Schedule' }, { value: 'webhook', label: 'Webhook' }, { value: 'new_data', label: 'New Data Available' }, { value: 'git_push', label: 'Git Push' }]} value={formData.type} onChange={e => setFormData({ ...formData, type: e.target.value })} />
          {formData.type === 'cron' && <Input label="Cron Expression" value={formData.config.cron_expression} onChange={e => setFormData({ ...formData, config: { ...formData.config, cron_expression: e.target.value } })} />}
          {formData.type === 'webhook' && <Input label="Webhook URL" value={formData.config.webhook_url || ''} onChange={e => setFormData({ ...formData, config: { ...formData.config, webhook_url: e.target.value } })} />}
          <Button onClick={() => createTrigger.mutate()} disabled={!formData.name}>Create</Button>
        </div>
      </Modal>
    </div>
  )
}