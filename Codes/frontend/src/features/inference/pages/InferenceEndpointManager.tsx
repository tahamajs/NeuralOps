// src/features/inference/pages/InferenceEndpointManager.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { inferenceAPI } from '../services/api'
import { EndpointCard } from '../components/EndpointCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { Server, Plus, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function InferenceEndpointManager({ onClose }: { onClose?: () => void }) {
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [formData, setFormData] = useState({ name: '', model_id: '', replicas: 1, max_batch_size: 32, timeout_ms: 5000, gpu: 0, cpu: 1, memory_mb: 1024 })
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: endpoints, isLoading, refetch } = useQuery({
    queryKey: ['inference-endpoints'],
    queryFn: () => inferenceAPI.getEndpoints(),
    refetchInterval: 10000,
  })

  const createEndpoint = useMutation({
    mutationFn: () => inferenceAPI.createEndpoint(formData),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['inference-endpoints'] })
      setShowCreateModal(false)
      setFormData({ name: '', model_id: '', replicas: 1, max_batch_size: 32, timeout_ms: 5000, gpu: 0, cpu: 1, memory_mb: 1024 })
      success('Endpoint created')
    },
  })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><Server size={20} className="text-neon-mint" /> Inference Endpoints</h1><p className="text-xs text-slate-400">Manage model serving endpoints</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => refetch()}><RefreshCw size={14} /></Button><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Endpoint</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid md:grid-cols-2 gap-4">{endpoints?.map(e => <EndpointCard key={e.id} endpoint={e} onRefresh={() => queryClient.invalidateQueries({ queryKey: ['inference-endpoints'] })} />)}</div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Create Inference Endpoint">
        <div className="space-y-4"><Input label="Name" value={formData.name} onChange={e => setFormData({ ...formData, name: e.target.value })} /><Input label="Model ID" value={formData.model_id} onChange={e => setFormData({ ...formData, model_id: e.target.value })} /><div className="grid grid-cols-2 gap-2"><Input label="Replicas" type="number" value={formData.replicas} onChange={e => setFormData({ ...formData, replicas: parseInt(e.target.value) })} /><Input label="GPU" type="number" value={formData.gpu} onChange={e => setFormData({ ...formData, gpu: parseInt(e.target.value) })} /></div><div className="grid grid-cols-2 gap-2"><Input label="CPU (cores)" type="number" value={formData.cpu} onChange={e => setFormData({ ...formData, cpu: parseFloat(e.target.value) })} /><Input label="Memory (MB)" type="number" value={formData.memory_mb} onChange={e => setFormData({ ...formData, memory_mb: parseInt(e.target.value) })} /></div><Button onClick={() => createEndpoint.mutate()} disabled={!formData.name || !formData.model_id}>Deploy</Button></div>
      </Modal>
    </div>
  )
}