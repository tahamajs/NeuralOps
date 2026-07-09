// src/features/agent/hooks/useAgentChat.ts
import { useEffect, useRef, useState, useCallback } from "react";
import api from "../../../api/client";
import { ChatMessage } from "../types";

export function useAgentChat(sessionId?: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [sending, setSending] = useState(false);
  const [wsStatus, setWsStatus] = useState<
    "connecting" | "open" | "closed" | "error"
  >("closed");
  const wsRef = useRef<WebSocket | null>(null);
  const initialLoadDone = useRef(false);

  // Load initial messages once
  useEffect(() => {
    if (!sessionId || initialLoadDone.current) return;
    initialLoadDone.current = true;
    setLoading(true);
    api
      .get(`/agents/chats/${sessionId}/messages/`)
      .then((res) => setMessages(res.data))
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [sessionId]);

  // Reset on session change
  useEffect(() => {
    initialLoadDone.current = false;
    setMessages([]);
  }, [sessionId]);

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const host =
      (import.meta as any).env.VITE_WS_URL || `${protocol}://${window.location.host}`;
    setWsStatus("connecting");
    const ws = new WebSocket(`${host}/ws/agent/${sessionId}/`);

    ws.onopen = () => setWsStatus("open");
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "messages_sync") {
          setMessages(data.messages);
          setLoading(false);
          setSending(false);
        }
      } catch (err) {
        console.error("WS parse error", err);
      }
    };
    ws.onerror = () => {
      setWsStatus("error");
      setSending(false);
      setLoading(false);
    };
    ws.onclose = () => {
      setWsStatus("closed");
      setSending(false);
    };

    wsRef.current = ws;
    return () => ws.close();
  }, [sessionId]);

  const sendMessage = useCallback(
    (text: string) => {
      if (!text.trim() || !sessionId) return;
      setSending(true);
      const optimisticMsg: ChatMessage = {
        id: `temp-${Date.now()}`,
        role: "user",
        content: text,
        created_at: new Date().toISOString(),
      };
      setMessages((prev) => [...prev, optimisticMsg]);

      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "message", text }));
      } else {
        setLoading(true);
        api
          .post(`/agents/chats/${sessionId}/send/`, { text })
          .then((res) => {
            setMessages(res.data.messages);
          })
          .catch(console.error)
          .finally(() => {
            setLoading(false);
            setSending(false);
          });
      }
    },
    [sessionId],
  );

  return { messages, loading, sending, wsStatus, sendMessage };
}
