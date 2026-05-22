import { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Button, Textarea, Spinner, Badge, Avatar } from '../../../components/ui'
import { VoteButtons } from '../components/VoteButtons'
import { MarkdownRenderer } from '../components/MarkdownRenderer'
import { Bookmark, CheckCircle, MessageCircle, Flag, Share2, Trash2, Edit, Lock, Award } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

export default function QuestionDetailPage() {
  const { questionId } = useParams()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [answerBody, setAnswerBody] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

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

  const deleteQuestion = useMutation({
    mutationFn: () => communityAPI.deleteQuestion(questionId!),
    onSuccess: () => navigate('/questions'),
  })

  if (isLoading) return <Spinner />

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
          <VoteButtons targetType="question" targetId={question.id} score={question.vote_score} userVote={question.user_vote} />
          <div className="flex-1">
            <MarkdownRenderer content={question.body} className="text-slate-300" />
            <div className="flex justify-between mt-6 pt-4 border-t border-white/5">
              <div className="flex gap-2">
                <button className="text-xs text-slate-500 hover:text-neon-blue flex items-center gap-1"><MessageCircle size={12} /> Add comment</button>
                <button className="text-xs text-slate-500 hover:text-neon-blue flex items-center gap-1"><Flag size={12} /> Report</button>
              </div>
              <div className="flex items-center gap-3">
                <Avatar src={question.author?.avatar_url} size="sm" />
                <div className="text-sm">
                  <div className="text-white font-medium">{question.author?.username}</div>
                  <div className="text-slate-500 text-xs">asked {formatDistanceToNow(new Date(question.created_at), { addSuffix: true })}</div>
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
            {answers.map((answer: any) => (
              <Card key={answer.id} className={answer.is_accepted ? 'border-green-500/30 bg-green-500/5' : ''}>
                <div className="flex gap-4">
                  <VoteButtons targetType="answer" targetId={answer.id} score={answer.vote_score} userVote={answer.user_vote} />
                  <div className="flex-1">
                    <MarkdownRenderer content={answer.body} className="text-slate-300" />
                    {answer.is_accepted && (
                      <div className="mt-2 inline-flex items-center gap-1 text-xs text-green-400"><CheckCircle size={12} /> Accepted Answer</div>
                    )}
                    <div className="flex justify-between mt-4 pt-2 border-t border-white/5">
                      <div className="flex gap-2">
                        <button className="text-xs text-slate-500 hover:text-neon-blue">Comment</button>
                        <button className="text-xs text-slate-500 hover:text-neon-blue">Edit</button>
                        <button className="text-xs text-slate-500 hover:text-red-400">Delete</button>
                      </div>
                      <div className="flex items-center gap-2 text-sm">
                        <span className="text-slate-500">answered {formatDistanceToNow(new Date(answer.created_at), { addSuffix: true })}</span>
                        <Avatar src={answer.author?.avatar_url} size="xs" />
                        <span className="text-white">{answer.author?.username}</span>
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
            {isSubmitting ? 'Posting...' : 'Post Answer'}
          </Button>
        </Card>
      )}
    </div>
  )
}