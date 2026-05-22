import { useChat } from '../hooks/useChat'
import { MessageList } from './MessageList'
import { MessageInput } from './MessageInput'
import { ConversationSidebar } from './ConversationSidebar'
import { TerminalCard } from '../../../components/ui/TerminalCard'

export function ChatContainer() {
  const { activeConversation, sendMessage, isConnected } = useChat()

  if (!activeConversation) {
    return <div className="flex items-center justify-center h-full">No active conversation</div>
  }

  return (
    <div className="flex h-full gap-4">
      <ConversationSidebar className="w-64 shrink-0" />
      <TerminalCard variant="glowing" noPadding className="flex flex-col flex-1 overflow-hidden">
        <div className="p-3 border-b border-white/10 flex justify-between items-center">
          <h3 className="text-sm font-medium text-white">{activeConversation.title}</h3>
          <div className="text-xs text-slate-500">{isConnected ? '● Connected' : '○ Offline'}</div>
        </div>
        <MessageList messages={activeConversation.messages} />
        <MessageInput onSendMessage={sendMessage} disabled={!isConnected} />
      </TerminalCard>
    </div>
  )
}