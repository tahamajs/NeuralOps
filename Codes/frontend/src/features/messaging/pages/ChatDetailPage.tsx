import { useParams } from 'react-router-dom'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Input, Button, Spinner } from '../../../components/ui'
import { useState, useEffect, useRef } from 'react'

export default function ChatDetailPage() {
  const { chatId } = useParams()
  const [message, setMessage] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()

  const { data: chat, isLoading } = useQuery({
    queryKey: ['chat', chatId],
    queryFn: async () => (await api.get(`/chats/${chatId}/`)).data,
  })
  const { data: messages } = useQuery({
    queryKey: ['chat-messages', chatId],
    queryFn: async () => (await api.get(`/chats/${chatId}/messages/`)).data,
    refetchInterval: 2000,
  })

  const send = useMutation({
    mutationFn: () => api.post(`/chats/${chatId}/send/`, { content: message }),
    onSuccess: () => { setMessage(''); queryClient.invalidateQueries({ queryKey: ['chat-messages', chatId] }) },
  })

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-[80vh]">
      <Card className="flex-1 overflow-y-auto p-4 space-y-3">
        {messages?.map((msg: any) => (
          <div key={msg.id} className={`flex ${msg.sender_username === chat?.current_user ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[70%] rounded-xl p-2 ${msg.sender_username === chat?.current_user ? 'bg-neon-blue/20' : 'bg-white/10'}`}>
              <p className="text-xs text-slate-400">{msg.sender_username}</p>
              <p className="text-white">{msg.content}</p>
              <p className="text-[10px] text-slate-500">{new Date(msg.created_at).toLocaleTimeString()}</p>
            </div>
          </div>
        ))}
        <div ref={messagesEndRef} />
      </Card>
      <div className="flex gap-2 mt-4">
        <Input value={message} onChange={e => setMessage(e.target.value)} onKeyDown={e => e.key === 'Enter' && send.mutate()} placeholder="Type a message..." />
        <Button onClick={() => send.mutate()}>Send</Button>
      </div>
    </div>
  )
}