import { useState, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Button, Textarea, Spinner, Badge, Avatar, Modal } from '../../../components/ui'
import { VoteButtons } from '../components/VoteButtons'
import { MarkdownRenderer } from '../components/MarkdownRenderer'
import { Bookmark, CheckCircle, MessageCircle, Flag, Share2, Trash2, Edit, Lock, Award, Send } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

export default function QuestionDetailPage() {
  const { questionId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [answerBody, setAnswerBody] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showBountyModal, setShowBountyModal] = useState(false)
  const [bountyAmount, setBountyAmount] = useState(50)

  const { data: question, isLoading } = useQuery({
    queryKey: ['question', questionId],
    queryFn: () => communityAPI.getQuestion(questionId!),
    enabled: !!questionId,
  })

  const { data: answers = [], isLoading: answersLoading } = useQuery({
    queryKey: ['answers', questionId],
    queryFn: () => communityAPI.getAnswers(questionId!),
    enabled: !!questionId,
  })

  const postAnswer = useMutation({
    mutationFn: (body: string) => communityAPI.postAnswer(questionId!, body),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['answers', questionId] })
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
      setAnswerBody('')
      setIsSubmitting(false)
    },
    onError: () => setIsSubmitting(false),
  })

  const bookmarkMutation = useMutation({
    mutationFn: () =>
      question?.bookmarked_by_current_user
        ? communityAPI.unbookmarkQuestion(questionId!)
        : communityAPI.bookmarkQuestion(questionId!),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['question', questionId] }),
  })

  const acceptAnswer = useMutation({
    mutationFn: (answerId: string) => communityAPI.acceptAnswer(answerId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['answers', questionId] })
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
    },
  })

  const startBounty = useMutation({
    mutationFn: (amount: number) => communityAPI.startBounty(questionId!, amount),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['question', questionId] })
      setShowBountyModal(false)
    },
  })

  const deleteQuestion = useMutation({
    mutationFn: () => communityAPI.deleteQuestion(questionId!),
    onSuccess: () => navigate('/questions'),
  })

  if (isLoading) return <Spinner />

  const sortedAnswers = [...answers].sort((a, b) => {
    if (a.is_accepted) return -1
    if (b.is_accepted) return 1
    return b.vote_score - a.vote_score
  })

  return (
    <div className="space-y-6 max-w-4xl mx-auto">
      {/* Question Header */}
      <div className="space-y-2">
        <div className="flex justify-between items-start">
          <h1 className="text-2xl font-bold text-white">{question.title}</h1>
          <div className="flex gap-2">
            <button onClick={() => bookmarkMutation.mutate()} className="p-2 rounded hover:bg-white/10">
              <Bookmark size={18} className={question.bookmarked_by_current_user ? 'text-yellow-400 fill-yellow-400' : 'text-slate-400'} />
            </button>
            <button className="p-2 rounded hover:bg-white/10"><Share2 size={18} className="text-slate-400" /></button>
            {question.is_closed && <Lock size={18} className="text-slate-500" />}
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          {question.tags?.map((tag: string) => <Badge key={tag}>#{tag}</Badge>)}
        </div>
      </div>

      {/* Question Body */}
      <Card>
        <div className="flex gap-4">
          <VoteButtons
            targetType="question"
            targetId={question.id}
            score={question.vote_score}
            userVote={question.user_vote}
          />
          <div className="flex-1">
            <MarkdownRenderer content={question.body} className="text-slate-300" />
            {question.bounty_amount && (
              <div className="mt-4 p-3 rounded-lg bg-yellow-500/10 border border-yellow-500/20">
                <div className="flex items-center gap-2 text-yellow-400">
                  <Award size={16} />
                  <span className="text-sm font-medium">Bounty of {question.bounty_amount} tokens</span>
                </div>
                {question.bounty_expires_at && (
                  <p className="text-xs text-slate-400 mt-1">
                    Expires {formatDistanceToNow(new Date(question.bounty_expires_at), { addSuffix: true })}
                  </p>
                )}
              </div>
            )}
            <div className="flex justify-between mt-6 pt-4 border-t border-white/5">
              <div className="flex gap-2">
                <button className="text-xs text-slate-500 hover:text-neon-blue flex items-center gap-1"><MessageCircle size={12} /> Add comment</button>
                <button className="text-xs text-slate-500 hover:text-neon-blue flex items-center gap-1"><Flag size={12} /> Report</button>
                {!question.bounty_amount && question.status === 'open' && (
                  <button onClick={() => setShowBountyModal(true)} className="text-xs text-yellow-400 hover:text-yellow-300 flex items-center gap-1">
                    <Award size={12} /> Start Bounty
                  </button>
                )}
              </div>
              <div className="flex items-center gap-3">
                <Avatar src={question.author?.avatar_url} size="sm" />
                <div className="text-sm">
                  <div className="text-white font-medium">{question.author?.username}</div>
                  <div className="text-slate-500 text-xs">
                    asked {formatDistanceToNow(new Date(question.created_at), { addSuffix: true })}
                    <span className="text-yellow-400 ml-2">● {question.author?.reputation} reputation</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Answers Section */}
      <div>
        <h2 className="text-lg font-bold text-white mb-4">
          {answers.length} {answers.length === 1 ? 'Answer' : 'Answers'}
        </h2>
        {answersLoading ? (
          <Spinner />
        ) : (
          <div className="space-y-4">
            {sortedAnswers.map((answer: Answer) => (
              <Card key={answer.id} className={answer.is_accepted ? 'border-green-500/30 bg-green-500/5' : ''}>
                <div className="flex gap-4">
                  <VoteButtons
                    targetType="answer"
                    targetId={answer.id}
                    score={answer.vote_score}
                    userVote={answer.user_vote}
                  />
                  <div className="flex-1">
                    <MarkdownRenderer content={answer.body} className="text-slate-300" />
                    {answer.is_accepted && (
                      <div className="mt-2 inline-flex items-center gap-1 text-xs text-green-400">
                        <CheckCircle size={12} /> Accepted Answer
                      </div>
                    )}
                    <div className="flex justify-between mt-4 pt-2 border-t border-white/5">
                      <div className="flex gap-2">
                        <button className="text-xs text-slate-500 hover:text-neon-blue">Comment</button>
                        <button className="text-xs text-slate-500 hover:text-neon-blue">Edit</button>
                        <button className="text-xs text-slate-500 hover:text-red-400">Delete</button>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-slate-500">
                          answered {formatDistanceToNow(new Date(answer.created_at), { addSuffix: true })}
                        </span>
                        <Avatar src={answer.author?.avatar_url} size="xs" />
                        <span className="text-white">{answer.author?.username}</span>
                        {!answer.is_accepted && question.author?.id === answer.author?.id && (
                          <Button size="sm" variant="ghost" onClick={() => acceptAnswer.mutate(answer.id)}>
                            <CheckCircle size={14} /> Accept
                          </Button>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </Card>
            ))}
          </div>
        )}
      </div>

      {/* Your Answer Form */}
      {!question?.is_closed && (
        <Card>
          <h3 className="text-sm font-bold text-white mb-3">Your Answer</h3>
          <Textarea rows={6} value={answerBody} onChange={(e) => setAnswerBody(e.target.value)} placeholder="Write your answer..." />
          <Button className="mt-3" onClick={() => { setIsSubmitting(true); postAnswer.mutate(answerBody); }} disabled={!answerBody.trim() || isSubmitting}>
            {isSubmitting ? <Spinner size="sm" /> : <Send size={14} className="mr-1" />} Post Answer
          </Button>
        </Card>
      )}

      {/* Bounty Modal */}
      <Modal open={showBountyModal} onClose={() => setShowBountyModal(false)} title="Start a Bounty">
        <div className="space-y-4">
          <p className="text-sm text-slate-400">
            Offering a bounty increases visibility and incentivizes high-quality answers.
          </p>
          <div>
            <label className="text-xs font-bold text-slate-400">Bounty Amount (tokens)</label>
            <Input type="number" value={bountyAmount} onChange={(e) => setBountyAmount(parseInt(e.target.value))} min={10} step={10} />
          </div>
          <Button onClick={() => startBounty.mutate(bountyAmount)} disabled={startBounty.isPending || bountyAmount < 10}>
            {startBounty.isPending ? 'Starting...' : 'Start Bounty'}
          </Button>
        </div>
      </Modal>
    </div>
  )
}