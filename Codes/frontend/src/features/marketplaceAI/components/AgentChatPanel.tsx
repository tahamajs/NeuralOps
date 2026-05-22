import { useState, useRef, useEffect } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Send, Loader2 } from 'lucide-react'

interface AgentChatPanelProps {
  onSendMessage: (content: string) => void
  isConnected: boolean
  streamingContent: string
  suggestions: string[]
}

export function AgentChatPanel({ onSendMessage, isConnected, streamingContent, suggestions }: AgentChatPanelProps) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState<{ role: 'user' | 'assistant'; content: string }[]>([])
  const messagesEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, streamingContent])

  const handleSend = () => { if (input.trim()) { setMessages(prev => [...prev, { role: 'user', content: input }]); onSendMessage(input); setInput('') } }

  useEffect(() => {
    if (streamingContent) {
      setMessages(prev => {
        const last = prev[prev.length - 1]
        if (last?.role === 'assistant') { const updated = [...prev]; updated[updated.length - 1] = { ...last, content: streamingContent }; return updated }
        return [...prev, { role: 'assistant', content: streamingContent }]
      })
    }
  }, [streamingContent])

  return (
    <TerminalCard className="flex flex-col h-full overflow-hidden"><div className="flex-1 overflow-y-auto p-3 space-y-3">{messages.map(msg => (<div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}><div className={`max-w-[85%] rounded-xl p-2 text-sm ${msg.role === 'user' ? 'bg-neon-blue/20 text-white' : 'bg-white/5 text-slate-300'}`}>{msg.content}</div></div>))}<div ref={messagesEndRef} /></div>
      <div className="border-t border-white/10 p-3"><div className="flex gap-2"><Input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} placeholder="Describe what you need..." disabled={!isConnected} className="flex-1" /><Button onClick={handleSend} disabled={!input.trim() || !isConnected}><Send size={16} /></Button></div></div>
    </TerminalCard>
  )
}