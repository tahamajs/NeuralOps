import { useEffect, useState, useRef, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";
import { useAuthStore } from "../../auth/store/authStore";

export function useLLMStream(sessionId?: string) {
  const [isStreaming, setIsStreaming] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const token = getAuthToken();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/playground/${sessionId || "temp"}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsStreaming(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "token") {
        setStreamingContent((prev) => prev + data.token);
      } else if (data.type === "error") {
        setError(data.message);
        setIsStreaming(false);
      } else if (data.type === "end") {
        setIsStreaming(false);
      }
    };
    ws.onerror = () => setError("WebSocket error");
    ws.onclose = () => setIsStreaming(false);
  }, [sessionId, token]);

  const sendMessage = useCallback(
    (messages: any[], config: any) => {
      setStreamingContent("");
      setError(null);
      if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
        connect();
        setTimeout(() => sendMessage(messages, config), 500);
        return;
      }
      wsRef.current.send(JSON.stringify({ type: "chat", messages, config }));
    },
    [connect],
  );

  const disconnect = useCallback(() => wsRef.current?.close(), []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return { isStreaming, streamingContent, error, sendMessage };
}
