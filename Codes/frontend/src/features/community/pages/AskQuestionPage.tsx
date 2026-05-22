import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import { communityAPI } from '../services/api'
import { Card, Input, Textarea, Button, Badge, Spinner } from '../../../components/ui'
import { X } from 'lucide-react'

export default function AskQuestionPage() {
  const navigate = useNavigate()
  const [title, setTitle] = useState('')
  const [body, setBody] = useState('')
  const [tags, setTags] = useState<string[]>([])
  const [tagInput, setTagInput] = useState('')
  const [isSubmitting, setIsSubmitting] = useState(false)

  const createQuestion = useMutation({
    mutationFn: () => communityAPI.createQuestion({ title, body, tags }),
    onSuccess: (data) => navigate(`/questions/${data.id}`),
    onError: () => setIsSubmitting(false),
  })

  const handleAddTag = () => {
    if (tagInput && !tags.includes(tagInput) && tags.length < 5) {
      setTags([...tags, tagInput])
      setTagInput('')
    }
  }

  const handleRemoveTag = (tag: string) => setTags(tags.filter(t => t !== tag))

  if (createQuestion.isError) return <div className="text-red-400">Failed to create question.</div>

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <h1 className="text-2xl font-bold text-white">Ask a Question</h1>
      <Card className="p-6">
        <div className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-white mb-1">Title</label>
            <p className="text-xs text-slate-400 mb-2">Be specific and imagine you're asking a question to another person</p>
            <Input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g., How to implement a custom training loop?" />
          </div>
          <div>
            <label className="block text-sm font-medium text-white mb-1">Body</label>
            <p className="text-xs text-slate-400 mb-2">Include all the information someone would need to answer your question</p>
            <Textarea rows={10} value={body} onChange={(e) => setBody(e.target.value)} placeholder="Describe your problem in detail..." />
          </div>
          <div>
            <label className="block text-sm font-medium text-white mb-1">Tags</label>
            <p className="text-xs text-slate-400 mb-2">Add up to 5 tags to describe what your question is about</p>
            <div className="flex gap-2 mb-2">
              <Input value={tagInput} onChange={(e) => setTagInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && handleAddTag()} placeholder="e.g., pytorch" />
              <Button size="sm" onClick={handleAddTag}>Add Tag</Button>
            </div>
            <div className="flex flex-wrap gap-2">
              {tags.map(tag => (
                <Badge key={tag} className="flex items-center gap-1">
                  {tag} <button onClick={() => handleRemoveTag(tag)}><X size={12} /></button>
                </Badge>
              ))}
            </div>
          </div>
          <Button className="w-full" onClick={() => { setIsSubmitting(true); createQuestion.mutate(); }} disabled={!title || !body || tags.length === 0 || isSubmitting}>
            {isSubmitting ? <Spinner size="sm" /> : 'Post Your Question'}
          </Button>
        </div>
      </Card>
    </div>
  )
}