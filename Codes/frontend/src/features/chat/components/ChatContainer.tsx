// src/features/chat/components/ChatContainer.tsx
import { useChat } from '../hooks/useChat';
import { MessageList } from './MessageList';
import { MessageInput } from './MessageInput';
import { ConversationSidebar } from './ConversationSidebar';
import { TerminalCard } from '../../../components/ui/TerminalCard';

export const ChatContainer = () => {
  const { messages, sendMessage, isConnected, activeConversation } = useChat();

  return (
    <div className="flex h-full gap-4">
      <ConversationSidebar className="w-64 shrink-0" />
      <TerminalCard variant="glowing" noPadding className="flex flex-col flex-1 overflow-hidden">
        <div className="p-3 border-b border-white/10">
          <h3 className="text-sm font-medium text-white">
            {activeConversation?.title || 'NeuralOps Assistant'}
          </h3>
          <div className="text-xs text-slate-500">
            {isConnected ? '● Connected' : '○ Connecting...'}
          </div>
        </div>
        <MessageList messages={messages} />
        <MessageInput onSendMessage={sendMessage} />
      </TerminalCard>
    </div>
  );
};