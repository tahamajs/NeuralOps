import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Input, Textarea, Button, Spinner } from '../../../components/ui'
import { TagInput } from '../components/TagInput'
import { MarkdownRenderer } from '../components/MarkdownRenderer'
import { Eye, EyeOff } from 'lucide-react'

export default function AskQuestionPage() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [isSubmitting, setIsSubmitting] = useState(false)
  const [showPreview, setShowPreview] = useState(false)

  const createQuestion = useMutation({
    mutationFn: () => communityAPI.createQuestion({ title, body, tags }),
    onSuccess: (data) => navigate(`/questions/${data.id}`),
    onError: () => setIsSubmitting(false),
  })

  const handleSubmit = () => {
    if (!title.trim() || !body.trim() || tags.length === 0) return
    setIsSubmitting(true)
    createQuestion.mutate()
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Ask a Question</h1>

      <Card className="p-6">
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-white mb-1">Title</label>
            <p className="text-xs text-slate-400 mb-2">Be specific and imagine you're asking a question to another person</p>
            <Input
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g., How to implement a custom training loop with PyTorch?"
            />
          </div>

          <div>
            <div className="flex justify-between items-center mb-1">
              <label className="block text-sm font-medium text-white">Body</label>
              <button
                type="button"
                onClick={() => setShowPreview(!showPreview)}
                className="text-xs text-slate-400 hover:text-neon-blue flex items-center gap-1"
              >
                {showPreview ? <EyeOff size={12} /> : <Eye size={12} />}
                {showPreview ? 'Hide Preview' : 'Show Preview'}
              </button>
            </div>
            <p className="text-xs text-slate-400 mb-2">Include all the information someone would need to answer your question</p>
            {showPreview ? (
              <div className="min-h-[300px] p-4 rounded-xl bg-white/5 border border-white/10">
                <MarkdownRenderer content={body} />
              </div>
            ) : (
              <Textarea
                rows={10}
                value={body}
                onChange={(e) => setBody(e.target.value)}
                placeholder="Describe your problem in detail... Use Markdown for formatting."
              />
            )}
          </div>

          <div>
            <label className="block text-sm font-medium text-white mb-1">Tags</label>
            <p className="text-xs text-slate-400 mb-2">Add up to 5 tags to describe what your question is about</p>
            <TagInput tags={tags} onChange={setTags} maxTags={5} />
          </div>

          <Button
            className="w-full"
            onClick={handleSubmit}
            disabled={!title || !body || tags.length === 0 || isSubmitting}
          >
            {isSubmitting ? <Spinner size="sm" /> : 'Post Your Question'}
          </Button>
        </div>
      </Card>
    </div>
  )
}