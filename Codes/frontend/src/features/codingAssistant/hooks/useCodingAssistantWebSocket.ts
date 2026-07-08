import { useEffect, useState, useRef, useCallback } from "react";

interface WebSocketMessage {
  type: "token" | "suggestion" | "error" | "complete";
  data: string | CodeSuggestion;
}

export function useCodingAssistantWebSocket(sessionId: string | null) {
  const [isConnected, setIsConnected] = useState(false);
  const [streamingContent, setStreamingContent] = useState("");
  const [suggestions, setSuggestions] = useState<CodeSuggestion[]>([]);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/code/${sessionId}/`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const msg: WebSocketMessage = JSON.parse(event.data);
      switch (msg.type) {
        case "token":
          setStreamingContent((prev) => prev + (msg.data as string));
          break;
        case "suggestion":
          setSuggestions((prev) => [...prev, msg.data as CodeSuggestion]);
          break;
        case "error":
          setError(msg.data as string);
          break;
        case "complete":
          setIsConnected(false);
          break;
      }
    };
    ws.onerror = () => setError("WebSocket connection error");
    ws.onclose = () => setIsConnected(false);
    return () => ws.close();
  }, [sessionId]);

  const sendMessage = useCallback((content: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "message", content }));
      setStreamingContent("");
      setSuggestions([]);
    }
  }, []);

  const applySuggestion = useCallback((suggestionId: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(
        JSON.stringify({ type: "apply", suggestion_id: suggestionId }),
      );
    }
  }, []);

  return {
    isConnected,
    streamingContent,
    suggestions,
    error,
    sendMessage,
    applySuggestion,
  };
}
