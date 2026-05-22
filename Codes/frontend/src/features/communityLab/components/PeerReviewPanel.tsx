// src/features/communityLab/components/PeerReviewPanel.tsx
import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { communityLabAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Textarea } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Badge } from '../../../components/ui/Badge'
import { CheckCircle, XCircle, Eye } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export function PeerReviewPanel({ labId }: { labId: string }) {
  const [selectedSubmission, setSelectedSubmission] = useState<any>(null)
  const [reviewComment, setReviewComment] = useState('')
  const [reviewScore, setReviewScore] = useState(3)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: submissions, isLoading } = useQuery({
    queryKey: ['peer-review-queue', labId],
    queryFn: () => communityLabAPI.getReviewQueue(labId),
  })

  const submitReview = useMutation({
    mutationFn: () => communityLabAPI.submitReview(labId, selectedSubmission.id, { comment: reviewComment, score: reviewScore, status: 'approved' }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['peer-review-queue', labId] }); setSelectedSubmission(null); setReviewComment(''); success('Review submitted') },
  })

  const rejectReview = useMutation({
    mutationFn: () => communityLabAPI.submitReview(labId, selectedSubmission.id, { comment: reviewComment, status: 'rejected' }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['peer-review-queue', labId] }); setSelectedSubmission(null); success('Submission rejected') },
  })

  if (isLoading) return <Spinner />

  return (
    <div className="grid lg:grid-cols-2 gap-4">{!selectedSubmission ? (<TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-3">Pending Review</h3><div className="space-y-2">{submissions?.filter(s => s.status === 'pending').map(sub => (<div key={sub.id} className="flex justify-between items-center p-2 rounded hover:bg-white/5"><div><div className="font-medium text-white">{sub.title}</div><div className="text-xs text-slate-400">Submitted by {sub.author.username}</div></div><Button size="sm" variant="ghost" onClick={() => setSelectedSubmission(sub)}><Eye size={14} /></Button></div>))}</div></TerminalCard>) : (<TerminalCard className="p-4"><div className="flex justify-between items-start"><div><h3 className="font-bold text-white">{selectedSubmission.title}</h3><p className="text-sm text-slate-400 mt-1">{selectedSubmission.description}</p></div><button onClick={() => setSelectedSubmission(null)} className="text-slate-400">✕</button></div><div className="mt-3"><div className="text-sm font-bold text-white mb-1">Your Review</div><Textarea rows={4} value={reviewComment} onChange={e => setReviewComment(e.target.value)} placeholder="Provide feedback..." /><div className="flex items-center gap-2 mt-2"><span className="text-sm text-slate-400">Score (1-5):</span><select value={reviewScore} onChange={e => setReviewScore(parseInt(e.target.value))} className="bg-white/5 border border-white/10 rounded p-1 text-sm"><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select></div><div className="flex gap-2 mt-4"><Button size="sm" variant="primary" onClick={() => submitReview.mutate()}><CheckCircle size={14} className="mr-1" /> Approve</Button><Button size="sm" variant="danger" onClick={() => rejectReview.mutate()}><XCircle size={14} className="mr-1" /> Reject</Button></div></div></TerminalCard>)}</div>
  )
}