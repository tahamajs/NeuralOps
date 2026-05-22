// src/features/chat/components/ChatPanel.tsx
import { useState, useRef, useEffect, useCallback } from 'react'
import { useResearchWebSocket } from '../hooks/useResearchWebSocket'
import { useChatStore } from '../store/chatStore'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter'
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Send, Loader2, User, Bot, AlertCircle } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

interface ChatPanelProps {
  sessionId?: string
  onClose?: () => void
}

export function ChatPanel({ sessionId: externalSessionId, onClose }: ChatPanelProps) {
  const [input, setInput] = useState('')
  const [isWaitingForConfirmation, setIsWaitingForConfirmation] = useState(false)
  const [pendingAction, setPendingAction] = useState<{ toolCall: any; toolId: string } | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const { success, error: toastError } = useToast()

  const {
    conversations,
    activeConversationId,
    addConversation,
    setActiveConversation,
    addMessage,
    updateLastAssistantMessage,
    getCurrentMessages,
  } = useChatStore()

  // Use external sessionId or the active one from store
  const currentSessionId = externalSessionId || activeConversationId
  const messages = currentSessionId ? getCurrentMessages(currentSessionId) : []

  const { status, reconnectCountdown, sendMessage, isConnected } = useResearchWebSocket(
    currentSessionId,
    {
      onMessage: (data) => {
        if (data.type === 'token') {
          // Streaming token – append to last assistant message
          updateLastAssistantMessage(currentSessionId!, data.token)
        } else if (data.type === 'tool_call') {
          // Tool call – show temporary message and ask for confirmation if destructive
          const isDestructive = data.tool_name === 'stop_run' || data.tool_name === 'delete_experiment'
          if (isDestructive) {
            setIsWaitingForConfirmation(true)
            setPendingAction({ toolCall: data, toolId: data.tool_call_id })
            addMessage(currentSessionId!, {
              id: `temp-${Date.now()}`,
              role: 'assistant',
              content: `⚠️ **The agent wants to execute:** ${data.tool_name}\n\n**Arguments:**\n\`\`\`json\n${JSON.stringify(data.arguments, null, 2)}\n\`\`\`\n\nDo you want to allow this action?`,
              createdAt: new Date(),
              isToolCall: true,
              toolCallId: data.tool_call_id,
            })
          } else {
            // Auto‑approve safe tools
            sendMessage({ type: 'tool_confirm', tool_call_id: data.tool_call_id, approved: true })
          }
        } else if (data.type === 'tool_result') {
          // Show tool result in a collapsible block
          addMessage(currentSessionId!, {
            id: `tool-${Date.now()}`,
            role: 'assistant',
            content: `**Tool result** (${data.tool_name}):\n\`\`\`\n${data.result.substring(0, 500)}${data.result.length > 500 ? '...' : ''}\n\`\`\``,
            createdAt: new Date(),
          })
        } else if (data.type === 'end') {
          // End of stream
          setIsWaitingForConfirmation(false)
          setPendingAction(null)
        } else if (data.type === 'error') {
          toastError('AI Error', data.message)
        }
      },
      onStatusChange: (status) => {
        if (status === 'open' && !currentSessionId) {
          // Create a new session automatically when connected
          const newId = addConversation('New Chat')
          setActiveConversation(newId)
        }
      },
    }
  )

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  const handleSendMessage = () => {
    if (!input.trim() || !isConnected) return
    // Add user message optimistically
    const userMessage = {
      id: `user-${Date.now()}`,
      role: 'user' as const,
      content: input,
      createdAt: new Date(),
    }
    addMessage(currentSessionId!, userMessage)
    sendMessage({ type: 'user_message', content: input })
    setInput('')
  }

  const handleConfirm = (approved: boolean) => {
    if (pendingAction && pendingAction.toolCall) {
      sendMessage({ type: 'tool_confirm', tool_call_id: pendingAction.toolCall.tool_call_id, approved })
    }
    setIsWaitingForConfirmation(false)
    setPendingAction(null)
  }

  const handleNewChat = () => {
    const newId = addConversation('New Chat')
    setActiveConversation(newId)
  }

  const handleSelectConversation = (id: string) => {
    setActiveConversation(id)
  }

  return (
    <div className="flex flex-col h-full bg-[#0a0c10]">
      {/* Header */}
      <div className="flex justify-between items-center p-3 border-b border-white/10 bg-black/30">
        <div className="flex items-center gap-2">
          <Bot size={20} className="text-neon-mint" />
          <h2 className="text-sm font-bold text-white">Research Assistant</h2>
          <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : status === 'connecting' ? 'bg-yellow-400' : 'bg-red-400'}`} />
          {!isConnected && status !== 'connecting' && reconnectCountdown > 0 && (
            <span className="text-xs text-slate-500">reconnecting in {reconnectCountdown}s</span>
          )}
        </div>
        <div className="flex gap-1">
          <button onClick={handleNewChat} className="p-1 rounded hover:bg-white/10 text-xs text-slate-400">+ New</button>
          {onClose && <button onClick={onClose} className="p-1 rounded hover:bg-white/10 text-slate-400">✕</button>}
        </div>
      </div>

      {/* Conversation sidebar (compact) */}
      <div className="flex border-b border-white/10 overflow-x-auto p-1 gap-1 bg-black/20">
        {conversations.map(conv => (
          <button
            key={conv.id}
            onClick={() => handleSelectConversation(conv.id)}
            className={`px-3 py-1 rounded text-xs whitespace-nowrap transition-colors ${
              activeConversationId === conv.id
                ? 'bg-neon-blue text-white'
                : 'text-slate-400 hover:bg-white/10'
            }`}
          >
            {conv.title}
          </button>
        ))}
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center text-slate-500 mt-8">
            <Bot size={40} className="mx-auto mb-2 opacity-30" />
            <p>Ask me about experiments, papers, or any research task.</p>
            <p className="text-xs mt-1">I can search papers, analyze logs, and even run experiments (with your approval).</p>
          </div>
        )}
        {messages.map(msg => (
          <div key={msg.id} className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
            <div className={`max-w-[85%] rounded-xl p-3 ${msg.role === 'user' ? 'bg-neon-blue/20 text-white' : 'bg-white/5 text-slate-300'}`}>
              {msg.role === 'assistant' ? (
                <ReactMarkdown
                  remarkPlugins={[remarkGfm]}
                  rehypePlugins={[rehypeHighlight]}
                  components={{
                    code({ node, inline, className, children, ...props }) {
                      const match = /language-(\w+)/.exec(className || '')
                      return !inline && match ? (
                        <SyntaxHighlighter
                          style={vscDarkPlus}
                          language={match[1]}
                          PreTag="div"
                          {...props}
                        >
                          {String(children).replace(/\n$/, '')}
                        </SyntaxHighlighter>
                      ) : (
                        <code className={className} {...props}>{children}</code>
                      )
                    },
                  }}
                >
                  {msg.content}
                </ReactMarkdown>
              ) : (
                <div className="whitespace-pre-wrap">{msg.content}</div>
              )}
              <div className="text-[10px] text-slate-500 mt-1 text-right">
                {new Date(msg.createdAt).toLocaleTimeString()}
              </div>
            </div>
          </div>
        ))}
        {isWaitingForConfirmation && (
          <div className="flex justify-start">
            <div className="bg-yellow-500/10 border border-yellow-500/30 rounded-xl p-3 max-w-[85%]">
              <p className="text-sm text-yellow-400 mb-2">⚠️ Destructive action requires your approval</p>
              <div className="flex gap-2">
                <Button size="sm" variant="danger" onClick={() => handleConfirm(false)}>Deny</Button>
                <Button size="sm" variant="primary" onClick={() => handleConfirm(true)}>Allow</Button>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="border-t border-white/10 p-3">
        <div className="flex gap-2">
          <Input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && !e.shiftKey && handleSendMessage()}
            placeholder="Ask me anything..."
            disabled={!isConnected || isWaitingForConfirmation}
            className="flex-1"
          />
          <Button
            onClick={handleSendMessage}
            disabled={!input.trim() || !isConnected || isWaitingForConfirmation}
          >
            <Send size={16} />
          </Button>
        </div>
        <div className="text-[10px] text-slate-500 mt-2 text-center">
          ⚡ The agent can search papers, analyze logs, stop runs (with confirmation), and create notes.
        </div>
      </div>
    </div>
  )
}