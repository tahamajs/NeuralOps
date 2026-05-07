import { useState, useRef, useEffect } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Badge } from '../../../components/ui/Badge'
import { Send, Bot, User, FlaskConical, Play, FileText, ChevronRight, Plus, Trash2 } from 'lucide-react'
import { formatDistanceToNow } from 'date-fns'

interface ChatMessage {
  id: string
  role: string
  content: string
  metadata?: any
  created_at: string
}

// Session sidebar component
function SessionSidebar({ sessions, activeId, onSelect, onNew }: {
  sessions: any[]
  activeId?: string
  onSelect: (id: string) => void
  onNew: () => void
}) {
  return (
    <div className="w-64 glass border-r border-white/5 flex flex-col h-full">
      <div className="p-3">
        <Button size="sm" className="w-full" onClick={onNew}>
          <Plus size={14} className="mr-1" /> New Chat
        </Button>
      </div>
      <div className="flex-1 overflow-y-auto scrollbar-thin px-2">
        {sessions.map(s => (
          <button
            key={s.id}
            onClick={() => onSelect(s.id)}
            className={`w-full text-left px-3 py-2 rounded-lg text-xs transition-colors mb-1 ${
              activeId === s.id ? 'bg-neon-blue/10 text-white border border-neon-blue/20' : 'text-slate-400 hover:bg-white/5'
            }`}
          >
            <div className="font-medium truncate">{s.title || 'New Chat'}</div>
            <div className="text-[10px] text-slate-500 mt-0.5">
              {formatDistanceToNow(new Date(s.updated_at), { addSuffix: true })}
            </div>
          </button>
        ))}
      </div>
    </div>
  )
}

// Main chat page
export default function AgentChatPage() {
  const { sessionId } = useParams<{ sessionId: string }>()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [showSidebar, setShowSidebar] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const wsRef = useRef<WebSocket | null>(null)

  // Fetch sessions list
  const { data: sessions = [] } = useQuery({
    queryKey: ['agentSessions'],
    queryFn: async () => (await api.get('/api/agents/chats/')).data,
  })

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = import.meta.env.VITE_WS_URL || `${protocol}://${window.location.host}`
    const ws = new WebSocket(`${host}/ws/agent/${sessionId}/`)

    ws.onopen = () => {
      // Load initial messages
      api.get(`/api/agents/chats/${sessionId}/messages/`).then(res => setMessages(res.data))
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'messages_sync') {
        setMessages(data.messages)
        setLoading(false)
      } else if (data.type === 'agent_response') {
        setLoading(false)
      }
    }

    ws.onerror = () => setLoading(false)
    ws.onclose = () => setLoading(false)

    wsRef.current = ws
    return () => ws.close()
  }, [sessionId])

  // Auto‑scroll
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSend = () => {
    if (!input.trim() || !sessionId) return
    setLoading(true)
    const text = input
    setInput('')

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'message', text }))
    } else {
      // REST fallback
      api.post(`/api/agents/chats/${sessionId}/send/`, { text }).then(res => {
        setMessages(res.data.messages)
        setLoading(false)
      }).catch(() => setLoading(false))
    }
  }

  const handleNewChat = async () => {
    const res = await api.post('/api/agents/chats/', { title: '' })
    navigate(`/agent/${res.data.id}`)
  }

  const handleSelectSession = (id: string) => {
    navigate(`/agent/${id}`)
  }

  // Render individual message
  const renderMessage = (msg: ChatMessage) => {
    // Experiment card (config draft or launch)
    if (msg.role === 'experiment_card') {
      const config = msg.metadata?.config || {}
      const type = msg.metadata?.type
      return (
        <div className="flex justify-center my-3">
          <Card className="w-full max-w-md border border-neon-blue/30 bg-neon-blue/5 animate-scale-in">
            <div className="flex items-center gap-2 mb-2">
              <FlaskConical size={18} className="text-neon-blue" />
              <span className="text-sm font-bold text-white">
                {type === 'experiment_launched' ? '🚀 Experiment Launched!' : 'Experiment Configuration'}
              </span>
            </div>
            <div className="text-xs space-y-1 text-slate-300">
              <p>Model: <span className="text-white">{config.model_name}</span></p>
              <p>Task: <span className="text-white">{config.task_name?.toUpperCase()}</span></p>
              <p>Attention: <span className="text-white">{config.attention_type || 'full'}</span></p>
              <p>Method: <span className="text-white">{config.method || 'baseline'}</span></p>
              <p>Epochs: {config.epochs} · Batch: {config.batch_size} · LR: {config.learning_rate}</p>
            </div>
            {config.run_id && (
              <Button size="sm" className="mt-2" onClick={() => navigate(`/experiments/${config.run_id}`)}>
                <Play size={14} className="mr-1" /> Monitor Experiment
              </Button>
            )}
          </Card>
        </div>
      )
    }

    // LaTeX preview card
    if (msg.metadata?.type === 'latex_preview') {
      return (
        <div className="flex justify-center my-3">
          <Card className="w-full max-w-md border border-green-500/30 bg-green-500/5">
            <div className="flex items-center gap-2 mb-2">
              <FileText size={18} className="text-green-400" />
              <span className="text-sm font-bold text-white">LaTeX Compilation</span>
            </div>
            <p className="text-xs text-slate-300">{msg.content.slice(0, 200)}</p>
            {msg.metadata?.pdf_url && (
              <Button size="sm" className="mt-2" onClick={() => window.open(msg.metadata.pdf_url, '_blank')}>
                View PDF
              </Button>
            )}
          </Card>
        </div>
      )
    }

    // Tool result (collapsed)
    if (msg.role === 'tool') {
      return (
        <div className="flex justify-start mb-2">
          <div className="bg-cyber-800/50 rounded-lg px-3 py-1.5 max-w-[80%] text-xs text-slate-500 font-mono">
            🔧 {msg.content.slice(0, 120)}...
          </div>
        </div>
      )
    }

    // Regular chat bubble
    const isUser = msg.role === 'user'
    return (
      <div className={`flex gap-2 mb-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
        {!isUser && (
          <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center flex-shrink-0">
            <Bot size={14} className="text-white" />
          </div>
        )}
        <div className={`max-w-[75%] rounded-2xl px-4 py-2.5 text-sm ${
          isUser
            ? 'bg-neon-blue/20 border border-neon-blue/30 text-white'
            : 'glass border border-white/10 text-slate-200'
        }`}>
          <div className="whitespace-pre-wrap">{msg.content}</div>
          <div className={`text-[10px] mt-1 ${isUser ? 'text-neon-blue/60' : 'text-slate-500'}`}>
            {new Date(msg.created_at).toLocaleTimeString()}
          </div>
        </div>
        {isUser && (
          <div className="w-7 h-7 rounded-full bg-green-500 flex items-center justify-center flex-shrink-0">
            <User size={14} className="text-white" />
          </div>
        )}
      </div>
    )
  }

  return (
    <div className="h-[calc(100vh-120px)] flex gap-0 max-w-6xl mx-auto">
      {/* Sidebar toggle for mobile */}
      <button
        className="lg:hidden fixed bottom-4 left-4 z-50 glass p-3 rounded-full shadow-lg"
        onClick={() => setShowSidebar(!showSidebar)}
      >
        <ChevronRight size={20} />
      </button>

      {/* Sidebar */}
      <div className={`${showSidebar ? 'block' : 'hidden'} lg:block fixed lg:relative z-40 h-full`}>
        <SessionSidebar
          sessions={sessions}
          activeId={sessionId}
          onSelect={handleSelectSession}
          onNew={handleNewChat}
        />
      </div>

      {/* Chat area */}
      <div className="flex-1 flex flex-col min-w-0">
        {/* Header */}
        <div className="glass px-4 py-3 mb-4 flex items-center gap-3">
          <Bot size={20} className="text-neon-blue" />
          <div>
            <h1 className="text-sm font-bold text-white">ML Training Assistant</h1>
            <p className="text-[10px] text-slate-400">
              I can search datasets, configure experiments, launch training, and edit LaTeX papers.
            </p>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto scrollbar-thin px-2 space-y-1">
          {messages.length === 0 && !loading && (
            <div className="flex items-center justify-center h-full text-slate-500 text-sm">
              <div className="text-center">
                <Bot size={48} className="mx-auto mb-4 text-slate-600" />
                <p>Ask me to help with your experiment!</p>
                <p className="text-xs mt-2 text-slate-600">
                  Try: "Fine-tune distilbert on sst2 with LoRA"<br />
                  "Search for text classification models"<br />
                  "Edit my LaTeX paper project abc123"
                </p>
              </div>
            </div>
          )}
          {messages.map(msg => (
            <div key={msg.id}>{renderMessage(msg)}</div>
          ))}
          {loading && (
            <div className="flex gap-2 mb-3">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-neon-blue to-purple-500 flex items-center justify-center">
                <Bot size={14} className="text-white" />
              </div>
              <div className="glass rounded-2xl px-4 py-2.5">
                <div className="flex gap-1">
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0s' }} />
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }} />
                  <div className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: '0.4s' }} />
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="glass p-3 mt-4 flex gap-2">
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSend(); } }}
            placeholder="Describe what you want to do..."
            className="flex-1"
          />
          <Button onClick={handleSend} disabled={!input.trim() || loading}>
            <Send size={16} />
          </Button>
        </div>
      </div>
    </div>
  )
}