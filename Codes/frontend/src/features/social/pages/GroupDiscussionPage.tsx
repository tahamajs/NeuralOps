// src/features/social/pages/GroupDiscussionPage.tsx
import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { socialAPI } from '../services/api'
import { DiscussionPost } from '../components/DiscussionPost'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Textarea } from '../../../components/ui/Input'
import { Modal } from '../../../components/ui/Modal'
import { Spinner } from '../../../components/ui/Spinner'
import { MessageSquare, Plus, MessageCircle, Heart } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function GroupDiscussionPage({ onClose }: { onClose?: () => void }) {
  const { groupId } = useParams()
  const [selectedDiscussionId, setSelectedDiscussionId] = useState<string | null>(null)
  const [showCreateModal, setShowCreateModal] = useState(false)
  const [newTitle, setNewTitle] = useState('')
  const [newContent, setNewContent] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: discussions, isLoading: threadsLoading } = useQuery({
    queryKey: ['discussions', groupId],
    queryFn: () => socialAPI.getDiscussionThreads(groupId!),
    enabled: !!groupId,
  })

  const { data: discussion, isLoading: discussionLoading } = useQuery({
    queryKey: ['discussion', selectedDiscussionId],
    queryFn: () => socialAPI.getDiscussion(selectedDiscussionId!),
    enabled: !!selectedDiscussionId,
  })

  const createDiscussion = useMutation({
    mutationFn: () => socialAPI.createDiscussion(groupId!, newTitle, newContent),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discussions', groupId] })
      setShowCreateModal(false)
      setNewTitle('')
      setNewContent('')
      success('Discussion created')
    },
  })

  const postReply = useMutation({
    mutationFn: (content: string) => socialAPI.postReply(selectedDiscussionId!, content),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discussion', selectedDiscussionId] })
      success('Reply posted')
    },
  })

  const likeDiscussion = useMutation({
    mutationFn: () => socialAPI.likeDiscussion(selectedDiscussionId!),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['discussion', selectedDiscussionId] })
    },
  })

  if (threadsLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white">Group Discussions</h1><p className="text-xs text-slate-400">Collaborate with lab members</p></div>
        <div className="flex gap-2"><Button size="sm" onClick={() => setShowCreateModal(true)}><Plus size={14} className="mr-1" /> New Discussion</Button>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4">
        <div className="grid lg:grid-cols-3 gap-4 h-full">
          <div className="space-y-2 overflow-auto">
            {discussions?.map(d => (
              <TerminalCard key={d.id} className={`p-3 cursor-pointer ${selectedDiscussionId === d.id ? 'border-neon-blue' : ''}`} onClick={() => setSelectedDiscussionId(d.id)}>
                <div className="flex justify-between"><h3 className="font-bold text-white">{d.title}</h3><span className="text-xs text-slate-400">{new Date(d.created_at).toLocaleDateString()}</span></div>
                <div className="flex gap-3 mt-1 text-xs text-slate-500"><span><MessageCircle size={12} className="inline mr-0.5" /> {d.reply_count}</span><span><Heart size={12} className="inline mr-0.5" /> {d.like_count}</span><span>by {d.author}</span></div>
              </TerminalCard>
            ))}
          </div>
          <div className="lg:col-span-2 overflow-auto">
            {discussionLoading ? <Spinner /> : discussion ? (
              <div className="space-y-4">
                <TerminalCard className="p-4">
                  <h2 className="text-xl font-bold text-white">{discussion.title}</h2>
                  <div className="flex justify-between items-center mt-1"><span className="text-sm text-slate-400">Posted by {discussion.author} · {new Date(discussion.created_at).toLocaleString()}</span><button onClick={() => likeDiscussion.mutate()} className={`flex items-center gap-1 text-sm ${discussion.user_liked ? 'text-red-400' : 'text-slate-400'}`}><Heart size={14} /> {discussion.like_count}</button></div>
                  <div className="mt-3 text-slate-300 whitespace-pre-wrap">{discussion.content}</div>
                </TerminalCard>
                <div className="space-y-2">
                  {discussion.replies.map(reply => <DiscussionPost key={reply.id} reply={reply} />)}
                </div>
                <TerminalCard className="p-4">
                  <Textarea rows={3} placeholder="Write a reply..." onKeyDown={(e) => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); postReply.mutate(e.currentTarget.value); e.currentTarget.value = '' } }} />
                  <Button size="sm" className="mt-2" onClick={(e) => { const textarea = e.currentTarget.previousSibling as HTMLTextAreaElement; postReply.mutate(textarea.value); textarea.value = '' }}>Post Reply</Button>
                </TerminalCard>
              </div>
            ) : <div className="text-center text-slate-500 py-12">Select a discussion to view</div>}
          </div>
        </div>
      </div>
      <Modal open={showCreateModal} onClose={() => setShowCreateModal(false)} title="Start New Discussion">
        <div className="space-y-4"><Input label="Title" value={newTitle} onChange={e => setNewTitle(e.target.value)} /><Textarea label="Content" rows={6} value={newContent} onChange={e => setNewContent(e.target.value)} /><Button onClick={() => createDiscussion.mutate()} disabled={!newTitle}>Post</Button></div>
      </Modal>
    </div>
  )
}