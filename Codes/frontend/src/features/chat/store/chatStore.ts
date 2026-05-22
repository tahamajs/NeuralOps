import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  createdAt: Date
  isLoading?: boolean
}

interface Conversation {
  id: string
  title: string
  messages: Message[]
  createdAt: Date
  updatedAt: Date
}

interface ChatState {
  conversations: Conversation[]
  activeConversationId: string | null
  isConnected: boolean
  addConversation: (title?: string) => string
  setActiveConversation: (id: string) => void
  addMessage: (conversationId: string, message: Message) => void
  updateMessage: (conversationId: string, messageId: string, content: string) => void
  removeConversation: (id: string) => void
  setIsConnected: (connected: boolean) => void
}

export const useChatStore = create<ChatState>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,
      isConnected: false,
      addConversation: (title = 'New Chat') => {
        const id = crypto.randomUUID()
        const newConv: Conversation = {
          id,
          title,
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        }
        set((state) => ({
          conversations: [newConv, ...state.conversations],
          activeConversationId: id,
        }))
        return id
      },
      setActiveConversation: (id) => set({ activeConversationId: id }),
      addMessage: (conversationId, message) => {
        set((state) => {
          const conv = state.conversations.find(c => c.id === conversationId)
          if (!conv) return state
          conv.messages.push(message)
          conv.updatedAt = new Date()
          if (conv.messages.length === 1 && message.role === 'user') {
            conv.title = message.content.slice(0, 30) + (message.content.length > 30 ? '...' : '')
          }
          return { conversations: [...state.conversations] }
        })
      },
      updateMessage: (conversationId, messageId, content) => {
        set((state) => {
          const conv = state.conversations.find(c => c.id === conversationId)
          if (!conv) return state
          const msg = conv.messages.find(m => m.id === messageId)
          if (msg) msg.content = content
          return { conversations: [...state.conversations] }
        })
      },
      removeConversation: (id) => {
        set((state) => ({
          conversations: state.conversations.filter(c => c.id !== id),
          activeConversationId: state.activeConversationId === id ? null : state.activeConversationId,
        }))
      },
      setIsConnected: (connected) => set({ isConnected: connected }),
    }),
    { name: 'chat-store' }
  )
)