import { useState, useRef, useEffect } from 'react'
import { useChatStore } from '../store/chatStore'
import { useToast } from '../../../components/ui/Toast'
import api from '../../../api/client'
import { Send, Plus } from 'lucide-react'
import ReactMarkdown from 'react-markdown'

export function ChatPanel({ onClose }: { onClose?: () => void }) {
  const { conversations, activeId, addConversation, addMessage, setActive } = useChatStore()
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { success, error } = useToast()

  const activeConv = conversations.find(c => c.id === activeId)

  useEffect(() => {
    if (messagesEndRef.current) messagesEndRef.current.scrollIntoView({ behavior: 'smooth' })
  }, [activeConv?.messages])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return
    if (!activeId) addConversation()
    const convId = activeId || conversations[0]?.id
    if (!convId) return

    const userMsg = { id: crypto.randomUUID(), role: 'user', content: input, createdAt: new Date() }
    addMessage(convId, userMsg)
    setInput('')
    setIsLoading(true)

    try {
      // Call your real AI endpoint here
      const res = await api.post('/chat/send/', { conversation_id: convId, message: input })
      const assistantMsg = { id: crypto.randomUUID(), role: 'assistant', content: res.data.reply, createdAt: new Date() }
      addMessage(convId, assistantMsg)
      success('AI responded')
    } catch (err) {
      error('Failed to get AI response')
    } finally {
      setIsLoading(false)
    }
  }

  const createNoteFromLast = async () => {
    const lastMsg = activeConv?.messages.slice(-1)[0]
    if (!lastMsg || lastMsg.role !== 'assistant') return
    try {
      await api.post('/notes/v1/notes/', { title: 'AI Note', content: lastMsg.content })
      success('Note created from AI response')
    } catch {
      error('Could not create note')
    }
  }

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-3 border-b border-white/10">
        <div className="flex gap-2">
          <button onClick={addConversation} className="p-1 rounded hover:bg-white/10"><Plus size={16} /></button>
          {conversations.map(c => (
            <button key={c.id} onClick={() => setActive(c.id)} className={`px-2 py-1 text-xs rounded ${activeId === c.id ? 'bg-neon-mint/20 text-neon-mint' : 'text-slate-400'}`}>
              {c.title}
            </button>
          ))}
        </div>
        {onClose && <button onClick={onClose} className="text-slate-400">✕</button>}
      </div>
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {activeConv?.messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[80%] rounded-lg p-3 ${msg.role === 'user' ? 'bg-neon-mint/10 text-white' : 'bg-white/5 text-slate-300'}`}>
              <ReactMarkdown className="prose prose-invert prose-sm">{msg.content}</ReactMarkdown>
              <div className="text-[10px] text-slate-500 mt-1">{new Date(msg.createdAt).toLocaleTimeString()}</div>
            </div>
          </div>
        ))}
        {isLoading && <div className="text-slate-400">AI is thinking...</div>}
        <div ref={messagesEndRef} />
      </div>
      <div className="p-3 border-t border-white/10 flex gap-2">
        <textarea
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && !e.shiftKey && sendMessage()}
          placeholder="Ask me anything, or type 'create note from last'"
          className="flex-1 bg-black/30 border border-white/10 rounded px-3 py-2 text-sm resize-none"
          rows={2}
        />
        <button onClick={sendMessage} disabled={isLoading} className="px-3 py-2 bg-neon-mint/20 rounded hover:bg-neon-mint/30">
          <Send size={16} />
        </button>
        <button onClick={createNoteFromLast} className="px-2 py-2 bg-white/5 rounded hover:bg-white/10" title="Create note from last AI response">📝</button>
      </div>
    </div>
  )
}