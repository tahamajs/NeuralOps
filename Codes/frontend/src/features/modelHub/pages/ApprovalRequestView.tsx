// src/features/modelHub/pages/ApprovalRequestView.tsx
import { useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { modelHubAPI } from '../services/api'
import { ApprovalTimeline } from '../components/ApprovalTimeline'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Textarea } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { CheckCircle, XCircle, AlertCircle, ArrowLeft } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function ApprovalRequestView({ onClose }: { onClose?: () => void }) {
  const { requestId } = useParams()
  const navigate = useNavigate()
  const [comment, setComment] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: request, isLoading } = useQuery({
    queryKey: ['approval-request', requestId],
    queryFn: () => modelHubAPI.getApprovalRequest(requestId!),
    enabled: !!requestId,
  })

  const reviewMutation = useMutation({
    mutationFn: ({ action, comment }: { action: ApprovalAction; comment?: string }) =>
      modelHubAPI.submitApproval(requestId!, action, comment),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['approval-request', requestId] })
      queryClient.invalidateQueries({ queryKey: ['approval-requests'] })
      success('Review submitted')
      navigate('/model-hub/approvals')
    },
  })

  if (isLoading) return <Spinner />
  if (!request) return <div className="p-4 text-red-400">Request not found</div>

  const isReviewer = request.reviewers.includes('current_user') // Replace with actual auth
  const isPending = request.status === 'pending'

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div className="flex items-center gap-3">
          <Button variant="ghost" size="sm" onClick={() => navigate(-1)}><ArrowLeft size={16} /></Button>
          <div><h1 className="text-xl font-bold text-white">Approval Request</h1><p className="text-xs text-slate-400">{request.model_name} v{request.model_version}</p></div>
        </div>
        {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <TerminalCard className="p-4">
          <div className="flex justify-between items-start">
            <div><div className="text-sm text-slate-400">Requested by</div><div className="text-white">{request.requested_by}</div><div className="text-xs text-slate-500 mt-1">{new Date(request.requested_at).toLocaleString()}</div></div>
            <Badge variant={request.status === 'pending' ? 'warning' : request.status === 'approved' ? 'success' : 'danger'}>{request.status}</Badge>
          </div>
        </TerminalCard>
        <ApprovalTimeline comments={request.comments} />
        {isPending && isReviewer && (
          <TerminalCard className="p-4">
            <h3 className="text-sm font-bold text-white mb-2">Your Review</h3>
            <Textarea rows={3} value={comment} onChange={e => setComment(e.target.value)} placeholder="Add comments..." className="mb-3" />
            <div className="flex gap-2">
              <Button variant="primary" onClick={() => reviewMutation.mutate({ action: 'approve', comment })}><CheckCircle size={14} className="mr-1" /> Approve</Button>
              <Button variant="danger" onClick={() => reviewMutation.mutate({ action: 'reject', comment })}><XCircle size={14} className="mr-1" /> Reject</Button>
              <Button variant="ghost" onClick={() => reviewMutation.mutate({ action: 'request_changes', comment })}><AlertCircle size={14} className="mr-1" /> Request Changes</Button>
            </div>
          </TerminalCard>
        )}
      </div>
    </div>
  )
}