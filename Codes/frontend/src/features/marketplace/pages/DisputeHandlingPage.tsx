// src/features/marketplace/pages/DisputeHandlingPage.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { marketplaceAPI } from '../services/api'
import { DisputeCard } from '../components/DisputeCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Textarea } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { AlertCircle, Plus, MessageSquare, CheckCircle } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function DisputeHandlingPage({ onClose }: { onClose?: () => void }) {
  const [statusFilter, setStatusFilter] = useState('')
  const [showRaiseModal, setShowRaiseModal] = useState(false)
  const [jobId, setJobId] = useState('')
  const [reason, setReason] = useState('')
  const [description, setDescription] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: disputes, isLoading } = useQuery({
    queryKey: ['disputes', statusFilter],
    queryFn: () => marketplaceAPI.getDisputes({ status: statusFilter || undefined }),
    refetchInterval: 10000,
  })

  const raiseDispute = useMutation({
    mutationFn: () => marketplaceAPI.raiseDispute(jobId, reason, description),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['disputes'] })
      setShowRaiseModal(false)
      setJobId('')
      setReason('')
      setDescription('')
      success('Dispute raised')
    },
  })

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><AlertCircle size={20} className="text-yellow-400" /> Dispute Center</h1><p className="text-xs text-slate-400">Resolve conflicts with consumers/providers</p></div>
        <div className="flex gap-2"><Select options={[{ value: '', label: 'All' }, { value: 'open', label: 'Open' }, { value: 'in_review', label: 'In Review' }, { value: 'resolved', label: 'Resolved' }]} value={statusFilter} onChange={e => setStatusFilter(e.target.value)} className="w-36" /><Button size="sm" onClick={() => setShowRaiseModal(true)}><Plus size={14} className="mr-1" /> Raise Dispute</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-3">
        {disputes?.length ? disputes.map(d => <DisputeCard key={d.id} dispute={d} onRefresh={() => queryClient.invalidateQueries({ queryKey: ['disputes'] })} />) : <div className="text-center text-slate-500 py-12">No disputes found</div>}
      </div>
      <Modal open={showRaiseModal} onClose={() => setShowRaiseModal(false)} title="Raise a Dispute">
        <div className="space-y-4"><Input label="Job ID" value={jobId} onChange={e => setJobId(e.target.value)} /><Select label="Reason" options={[{ value: 'service_not_delivered', label: 'Service not delivered' }, { value: 'incorrect_billing', label: 'Incorrect billing' }, { value: 'technical_issue', label: 'Technical issue' }, { value: 'other', label: 'Other' }]} value={reason} onChange={e => setReason(e.target.value)} /><Textarea label="Description" rows={4} value={description} onChange={e => setDescription(e.target.value)} placeholder="Provide details..." /><Button onClick={() => raiseDispute.mutate()} disabled={!jobId || !reason}>Submit Dispute</Button></div>
      </Modal>
    </div>
  )
}