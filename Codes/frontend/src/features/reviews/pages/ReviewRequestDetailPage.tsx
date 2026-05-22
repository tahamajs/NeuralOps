import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button, Spinner, Badge, Textarea } from '../../../components/ui'
import { useState } from 'react'

export default function ReviewRequestDetailPage() {
  const { requestId } = useParams()
  const queryClient = useQueryClient()
  const [reviewText, setReviewText] = useState('')
  const [comment, setComment] = useState('')

  const { data: request, isLoading } = useQuery({
    queryKey: ['review-request', requestId],
    queryFn: async () => (await api.get(`/requests/${requestId}/`)).data,
  })
  const { data: reviews } = useQuery({
    queryKey: ['reviews', requestId],
    queryFn: async () => (await api.get(`/requests/${requestId}/review/`)).data,
  })
  const { data: comments } = useQuery({
    queryKey: ['review-comments', requestId],
    queryFn: async () => (await api.get(`/reviews/${request?.review_id}/comment/`)).data,
    enabled: !!request?.review_id,
  })

  const submitReview = useMutation({
    mutationFn: () => api.post(`/requests/${requestId}/review/`, { content: reviewText }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['reviews', requestId] }),
  })
  const addComment = useMutation({
    mutationFn: () => api.post(`/reviews/${request?.review_id}/comment/`, { content: comment }),
    onSuccess: () => { queryClient.invalidateQueries({ queryKey: ['review-comments', requestId] }); setComment('') },
  })
  const voteHelpful = useMutation({
    mutationFn: (reviewId: string) => api.post(`/reviews/${reviewId}/vote/`, { helpful: true }),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['reviews', requestId] }),
  })

  if (isLoading) return <Spinner />

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      <Card>
        <h1 className="text-xl font-bold text-white">{request?.title}</h1>
        <div className="text-sm text-slate-400 mt-1">Requested by {request?.requester_username}</div>
        <div className="prose prose-invert mt-4" dangerouslySetInnerHTML={{ __html: request?.description }} />
      </Card>

      {request?.status === 'open' && (
        <Card>
          <h2 className="text-sm font-bold text-white">Submit Your Review</h2>
          <Textarea rows={6} value={reviewText} onChange={e => setReviewText(e.target.value)} placeholder="Write your review..." />
          <Button className="mt-2" onClick={() => submitReview.mutate()}>Submit Review</Button>
        </Card>
      )}

      {reviews?.length > 0 && (
        <Card>
          <h2 className="text-sm font-bold text-white">Reviews</h2>
          {reviews.map((rev: any) => (
            <div key={rev.id} className="border-t border-white/5 py-3">
              <div className="flex justify-between">
                <span className="font-semibold text-white">{rev.author_username}</span>
                <Button size="sm" variant="ghost" onClick={() => voteHelpful.mutate(rev.id)}>👍 Helpful ({rev.helpful_count})</Button>
              </div>
              <div className="text-sm text-slate-300 mt-1">{rev.content}</div>
            </div>
          ))}
        </Card>
      )}

      <Card>
        <h2 className="text-sm font-bold text-white">Comments</h2>
        {comments?.map((c: any) => (
          <div key={c.id} className="text-sm border-b border-white/5 py-2"><span className="text-neon-blue">{c.author_username}:</span> {c.content}</div>
        ))}
        <div className="mt-3 flex gap-2">
          <Textarea rows={2} value={comment} onChange={e => setComment(e.target.value)} placeholder="Add a comment..." />
          <Button onClick={() => addComment.mutate()}>Post</Button>
        </div>
      </Card>
    </div>
  )
}