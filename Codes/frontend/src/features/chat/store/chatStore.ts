// src/features/chat/store/chatStore.ts
import { create } from "zustand";
import { persist } from "zustand/middleware";

const uuid = (): string => {
  if (
    typeof crypto !== "undefined" &&
    typeof crypto.randomUUID === "function"
  ) {
    return crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = (Math.random() * 16) | 0;
    const v = c === "x" ? r : (r & 0x3) | 0x8;
    return v.toString(16);
  });
};

export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  createdAt: Date;
  isToolCall?: boolean;
  toolCallId?: string;
}

export interface Conversation {
  id: string;
  title: string;
  messages: ChatMessage[];
  createdAt: Date;
  updatedAt: Date;
}

interface ChatStore {
  conversations: Conversation[];
  activeConversationId: string | null;
  addConversation: (title?: string) => string;
  setActiveConversation: (id: string) => void;
  addMessage: (convId: string, msg: ChatMessage) => void;
  updateLastAssistantMessage: (convId: string, token: string) => void;
  getCurrentMessages: (convId: string) => ChatMessage[];
  deleteConversation: (id: string) => void;
}

export const useChatStore = create<ChatStore>()(
  persist(
    (set, get) => ({
      conversations: [],
      activeConversationId: null,

      addConversation: (title = "New Chat") => {
        const id = uuid();
        const newConv: Conversation = {
          id,
          title,
          messages: [],
          createdAt: new Date(),
          updatedAt: new Date(),
        };
        set((state) => ({
          conversations: [newConv, ...state.conversations],
          activeConversationId: id,
        }));
        return id;
      },

      setActiveConversation: (id) => set({ activeConversationId: id }),

      addMessage: (convId, msg) => {
        set((state) => {
          const conv = state.conversations.find((c) => c.id === convId);
          if (!conv) return state;
          conv.messages.push(msg);
          conv.updatedAt = new Date();
          if (conv.messages.length === 1 && msg.role === "user") {
            conv.title =
              msg.content.slice(0, 30) + (msg.content.length > 30 ? "..." : "");
          }
          return { conversations: [...state.conversations] };
        });
      },

      updateLastAssistantMessage: (convId, token) => {
        set((state) => {
          const conv = state.conversations.find((c) => c.id === convId);
          if (!conv) return state;
          const lastMsg = conv.messages[conv.messages.length - 1];
          if (lastMsg && lastMsg.role === "assistant") {
            lastMsg.content += token;
            conv.updatedAt = new Date();
          } else {
            // If no assistant message, create one
            conv.messages.push({
              id: `assistant-${Date.now()}`,
              role: "assistant",
              content: token,
              createdAt: new Date(),
            });
          }
          return { conversations: [...state.conversations] };
        });
      },

      getCurrentMessages: (convId) => {
        const conv = get().conversations.find((c) => c.id === convId);
        return conv?.messages || [];
      },

      deleteConversation: (id) => {
        set((state) => ({
          conversations: state.conversations.filter((c) => c.id !== id),
          activeConversationId:
            state.activeConversationId === id
              ? null
              : state.activeConversationId,
        }));
      },
    }),
    { name: "chat-store" },
  ),
);
