// src/features/modelHub/pages/PromotionPanel.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { modelHubAPI } from '../services/api'
import { StageBadge } from '../components/StageBadge'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { GitBranch, ArrowRight, CheckCircle, XCircle } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function PromotionPanel({ onClose }: { onClose?: () => void }) {
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: promotions, isLoading } = useQuery({
    queryKey: ['promotion-requests'],
    queryFn: () => modelHubAPI.getPromotionRequests(),
  })

  const approvePromotion = useMutation({
    mutationFn: (id: string) => modelHubAPI.approvePromotion(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promotion-requests'] })
      success('Promotion approved')
    },
  })

  const rejectPromotion = useMutation({
    mutationFn: (id: string) => modelHubAPI.rejectPromotion(id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['promotion-requests'] })
      success('Promotion rejected')
    },
  })

  if (isLoading) return <Spinner />

  const pending = promotions?.filter(p => p.status === 'pending') || []
  const history = promotions?.filter(p => p.status !== 'pending') || []

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><GitBranch size={20} className="text-neon-mint" /> Promotion Panel</h1><p className="text-xs text-slate-400">Manage model promotions across environments</p></div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-6">
        <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Pending Promotions ({pending.length})</h3>{pending.length === 0 ? <div className="text-slate-500 text-sm text-center py-4">No pending promotions</div> : pending.map(p => (<div key={p.id} className="flex justify-between items-center p-3 border-b border-white/5"><div><div className="font-medium text-white">Model ID: {p.model_id.slice(0, 8)}</div><div className="flex items-center gap-2 mt-1"><StageBadge stage={p.from_stage} /><ArrowRight size={12} className="text-slate-400" /><StageBadge stage={p.to_stage} /></div><div className="text-xs text-slate-400 mt-1">Requested by {p.requested_by} · {new Date(p.requested_at).toLocaleString()}</div></div><div className="flex gap-2"><Button size="sm" variant="primary" onClick={() => approvePromotion.mutate(p.id)}><CheckCircle size={12} className="mr-1" /> Approve</Button><Button size="sm" variant="danger" onClick={() => rejectPromotion.mutate(p.id)}><XCircle size={12} className="mr-1" /> Reject</Button></div></div>))}</TerminalCard>
        <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Promotion History</h3><div className="space-y-2">{history.map(p => (<div key={p.id} className="flex justify-between items-center p-2 border-b border-white/5"><div><div>Model {p.model_id.slice(0, 8)}: {p.from_stage} → {p.to_stage}</div><div className="text-xs text-slate-400">{p.status === 'approved' ? 'Approved by ' + p.approved_by : 'Rejected'} · {new Date(p.approved_at || p.requested_at).toLocaleString()}</div></div><Badge variant={p.status === 'approved' ? 'success' : 'danger'}>{p.status}</Badge></div>))}</div></TerminalCard>
      </div>
    </div>
  )
}