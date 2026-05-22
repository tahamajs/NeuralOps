// src/features/communityLab/components/ChatroomView.tsx
import { useState, useEffect, useRef } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { communityLabAPI } from '../services/api'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Spinner } from '../../../components/ui/Spinner'
import { Send, Plus, MessageSquare, Users } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export function ChatroomView({ labId }: { labId: string }) {
  const [activeRoomId, setActiveRoomId] = useState<string | null>(null)
  const [messageInput, setMessageInput] = useState('')
  const [messages, setMessages] = useState<any[]>([])
  const [ws, setWs] = useState<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: rooms, isLoading: roomsLoading } = useQuery({
    queryKey: ['chatrooms', labId],
    queryFn: () => communityLabAPI.getChatrooms(labId),
  })

  const createRoom = useMutation({
    mutationFn: (name: string) => communityLabAPI.createChatroom(labId, name),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['chatrooms', labId] }),
  })

  useEffect(() => {
    if (!activeRoomId) return
    const token = localStorage.getItem('token')
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/lab/${labId}/chat/${activeRoomId}/?token=${token}`
    const websocket = new WebSocket(wsUrl)
    websocket.onmessage = (event) => { const data = JSON.parse(event.data); setMessages(prev => [...prev, data]) }
    setWs(websocket)
    return () => websocket.close()
  }, [activeRoomId, labId])

  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  const sendMessage = () => {
    if (ws && messageInput.trim()) { ws.send(JSON.stringify({ type: 'message', content: messageInput })); setMessageInput('') }
  }

  if (roomsLoading) return <Spinner />

  return (
    <div className="grid lg:grid-cols-3 gap-4 h-[500px]">
      <TerminalCard className="p-3 overflow-y-auto"><div className="flex justify-between items-center mb-3"><h3 className="text-sm font-bold text-white">Rooms</h3><Button size="sm" variant="ghost" onClick={() => { const name = prompt('Room name:'); if (name) createRoom.mutate(name) }}><Plus size={14} /></Button></div>
        <div className="space-y-1">{rooms?.map(room => (<button key={room.id} onClick={() => setActiveRoomId(room.id)} className={`w-full text-left px-3 py-2 rounded text-sm ${activeRoomId === room.id ? 'bg-neon-blue/20 text-white' : 'text-slate-400 hover:bg-white/5'}`}><div className="flex items-center gap-2"><MessageSquare size={14} /><span>{room.name}</span></div><div className="text-xs text-slate-500">{room.member_count} members</div></button>))}</div></TerminalCard>
      <TerminalCard className="lg:col-span-2 flex flex-col overflow-hidden"><div className="flex-1 overflow-y-auto p-3 space-y-2">{messages.map((msg, idx) => (<div key={idx} className={`flex ${msg.user === 'current' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[70%] rounded-lg p-2 text-sm ${msg.user === 'current' ? 'bg-neon-blue/20 text-white' : 'bg-white/5 text-slate-300'}`}><div className="text-xs text-slate-400">{msg.username}</div>{msg.content}<div className="text-[10px] text-slate-500 mt-1">{new Date(msg.timestamp).toLocaleTimeString()}</div></div></div>))}<div ref={messagesEndRef} /></div>
        <div className="border-t border-white/10 p-3 flex gap-2"><Input value={messageInput} onChange={e => setMessageInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && sendMessage()} placeholder="Type a message..." className="flex-1" /><Button onClick={sendMessage}><Send size={16} /></Button></div>
      </TerminalCard>
    </div>
  )
}