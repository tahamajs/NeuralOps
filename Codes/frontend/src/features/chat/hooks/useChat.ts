import { useEffect, useRef, useCallback } from "react";
import { useChatStore } from "../store/chatStore";
import { useToast } from "../../../components/ui/Toast";

export function useChat() {
  const wsRef = useRef<WebSocket | null>(null);
  const {
    conversations,
    activeConversationId,
    addMessage,
    updateMessage,
    setIsConnected,
  } = useChatStore() as any;
  const { success, error: showError } = useToast();

  const activeConversation = conversations.find(
    (c) => c.id === activeConversationId,
  );

  const sendMessage = useCallback(
    async (content: string) => {
      if (!activeConversationId) return;
      const userMessage: Message = {
        id: crypto.randomUUID(),
        role: "user",
        content,
        createdAt: new Date(),
      };
      addMessage(activeConversationId, userMessage);

      const assistantMessageId = crypto.randomUUID();
      addMessage(activeConversationId, {
        id: assistantMessageId,
        role: "assistant",
        content: "",
        createdAt: new Date(),
        isLoading: true,
      });

      try {
        const token = localStorage.getItem("token");
        const ws = new WebSocket(
          `${(import.meta as any).env.VITE_WS_URL || "ws://localhost:8000"}/ws/chat/?token=${token}`,
        );
        ws.onopen = () => {
          ws.send(
            JSON.stringify({
              type: "message",
              conversationId: activeConversationId,
              content,
            }),
          );
        };
        ws.onmessage = (event) => {
          const data = JSON.parse(event.data);
          if (data.type === "chunk") {
            updateMessage(
              activeConversationId,
              assistantMessageId,
              (prev) => prev + data.chunk,
            );
          } else if (data.type === "done") {
            updateMessage(
              activeConversationId,
              assistantMessageId,
              (prev) => prev,
            );
            ws.close();
          } else if (data.type === "error") {
            showError("Chat error", data.message);
            ws.close();
          }
        };
        ws.onerror = () =>
          showError("Connection error", "Could not reach chat server");
        ws.onclose = () => setIsConnected(false);
      } catch (err) {
        showError("Failed to send message", String(err));
      }
    },
    [activeConversationId, addMessage, updateMessage, showError],
  );

  return {
    conversations,
    activeConversation,
    sendMessage,
    isConnected: true, // simplified for demo
  };
}
