import { useState, useRef, useEffect } from 'react'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Send, Loader2, Sparkles, FileCode, CheckCircle, XCircle } from 'lucide-react'
import { Message, CodeSuggestion } from '../types'
import ReactMarkdown from 'react-markdown'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'

interface ChatPanelProps {
  messages: Message[]
  onSendMessage: (content: string) => void
  isStreaming: boolean
  streamingContent: string
  suggestions: CodeSuggestion[]
  onApplySuggestion: (suggestionId: string) => void
  isConnected: boolean
}

export function ChatPanel({
  messages,
  onSendMessage,
  isStreaming,
  streamingContent,
  suggestions,
  onApplySuggestion,
  isConnected,
}: ChatPanelProps) {
  const [input, setInput] = useState('')
  const messagesEndRef = useRef<HTMLDivElement>(null)
  useEffect(() => { messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, streamingContent])

  const handleSend = () => { if (input.trim()) { onSendMessage(input); setInput('') } }

  return (
    <TerminalCard className="flex flex-col h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-xl p-3 ${msg.role === 'user' ? 'bg-neon-blue/20 text-white' : 'bg-white/5 text-slate-300'}`}>
              {msg.role === 'assistant' ? (
                <ReactMarkdown
                  components={{ code({ node, inline, className, children, ...props }) { const match = /language-(\w+)/.exec(className || ''); return !inline && match ? <SyntaxHighlighter style={vscDarkPlus} language={match[1]} PreTag="div" {...props}>{String(children).replace(/\n$/, '')}</SyntaxHighlighter> : <code className={className} {...props}>{children}</code> } }}
                >{msg.content}</ReactMarkdown>
              ) : <div className="whitespace-pre-wrap">{msg.content}</div>}
              <div className="text-[10px] text-slate-500 mt-1">{msg.timestamp.toLocaleTimeString()}</div>
            </div>
          </div>
        ))}
        {isStreaming && streamingContent && (
          <div className="flex justify-start"><div className="max-w-[85%] rounded-xl p-3 bg-white/5 text-slate-300"><ReactMarkdown>{streamingContent}</ReactMarkdown><div className="flex items-center gap-1 mt-2"><Loader2 size={12} className="animate-spin" /><span className="text-xs text-slate-500">Streaming...</span></div></div></div>
        )}
        {suggestions.length > 0 && (
          <div className="space-y-2 mt-2"><div className="text-xs font-bold text-neon-mint">💡 AI Suggestions</div>
            {suggestions.map(sug => (
              <div key={sug.id} className="bg-yellow-500/10 border border-yellow-500/30 rounded-lg p-3">
                <div className="flex justify-between items-start"><div><div className="font-medium text-white text-sm">{sug.title}</div><div className="text-xs text-slate-400 mt-0.5">{sug.description}</div></div>
                  <Button size="sm" variant="ghost" onClick={() => onApplySuggestion(sug.id)} disabled={sug.applied}>{sug.applied ? <CheckCircle size={14} className="text-green-400" /> : <Sparkles size={14} />} {!sug.applied && 'Apply'}</Button>
                </div>
                {sug.suggestedCode && <pre className="mt-2 text-xs bg-black/50 p-2 rounded overflow-auto"><code>{sug.suggestedCode}</code></pre>}
              </div>
            ))}
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>
      <div className="border-t border-white/10 p-3"><div className="flex gap-2"><Input value={input} onChange={e => setInput(e.target.value)} onKeyDown={e => e.key === 'Enter' && handleSend()} placeholder="Describe what you want to build or ask for help..." disabled={!isConnected} className="flex-1" /><Button onClick={handleSend} disabled={!input.trim() || !isConnected}><Send size={16} /></Button></div></div>
    </TerminalCard>
  )
}