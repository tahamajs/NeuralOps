import React, { useState, useEffect, useRef } from 'react'
import { Send, Loader2, Sparkles } from 'lucide-react'
import { useAuthStore } from '../features/auth/store/authStore'

export default function ResearchAgentChat({ sessionId }: { sessionId: string }) {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([])
  const [input, setInput] = useState('')
  const [streaming, setStreaming] = useState(false)
  const [currentStream, setCurrentStream] = useState('')
  const wsRef = useRef<WebSocket | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const token = useAuthStore(s => s.token)

  useEffect(() => {
    const wsUrl = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}/ws/agent/${sessionId}/?token=${token}`
    const ws = new WebSocket(wsUrl)
    wsRef.current = ws

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'token') {
        setCurrentStream(prev => prev + data.text)
      } else if (data.type === 'done') {
        setMessages(prev => [...prev, { role: 'assistant', content: currentStream }])
        setCurrentStream('')
        setStreaming(false)
      }
    }

    return () => ws.close()
  }, [sessionId, token])

  const sendMessage = () => {
    if (!input.trim() || streaming) return
    setMessages(prev => [...prev, { role: 'user', content: input }])
    setStreaming(true)
    setCurrentStream('')
    wsRef.current?.send(JSON.stringify({ type: 'message', text: input }))
    setInput('')
  }

  return (
    <div className="flex flex-col h-full bg-dark-bg">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((m, i) => (
          <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] p-3 rounded-xl ${m.role === 'user' ? 'bg-neon-blue text-white' : 'glass text-white'}`}>
              {m.content}
            </div>
          </div>
        ))}
        {streaming && currentStream && (
          <div className="flex justify-start">
            <div className="max-w-[80%] p-3 rounded-xl glass text-white">
              {currentStream}
              <span className="animate-pulse">▌</span>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-4 border-t border-white/10">
        <div className="flex gap-2">
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about your research..."
            className="flex-1 bg-white/5 border border-white/10 rounded-xl px-4 py-2 text-white"
            disabled={streaming}
          />
          <button onClick={sendMessage} disabled={streaming} className="bg-neon-blue p-2 rounded-xl">
            {streaming ? <Loader2 className="animate-spin" size={20} /> : <Send size={20} />}
          </button>
        </div>
      </div>
    </div>
  )
}