import { useChatStore } from '../store/chatStore'
import { cn } from '../../../lib/utils'
import { MessageSquare, Plus, Trash2 } from 'lucide-react'

export function ConversationSidebar({ className }: { className?: string }) {
  const { conversations, activeConversationId, addConversation, setActiveConversation, removeConversation } = useChatStore()

  return (
    <div className={cn("glass rounded-2xl p-2", className)}>
      <button
        onClick={() => addConversation()}
        className="w-full flex items-center gap-2 px-3 py-2 rounded-lg bg-neon-blue/20 text-neon-blue hover:bg-neon-blue/30 transition-colors mb-2"
      >
        <Plus size={16} /> New Chat
      </button>
      <div className="space-y-1">
        {conversations.map(conv => (
          <div
            key={conv.id}
            className={cn(
              "flex items-center justify-between px-3 py-2 rounded-lg cursor-pointer hover:bg-white/5 transition-colors",
              activeConversationId === conv.id && "bg-white/10"
            )}
            onClick={() => setActiveConversation(conv.id)}
          >
            <div className="flex items-center gap-2 truncate">
              <MessageSquare size={14} className="text-slate-400" />
              <span className="text-sm text-white truncate">{conv.title}</span>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); removeConversation(conv.id) }}
              className="opacity-0 group-hover:opacity-100 p-1 rounded hover:bg-white/10"
            >
              <Trash2 size={12} className="text-red-400" />
            </button>
          </div>
        ))}
      </div>
    </div>
  )
}