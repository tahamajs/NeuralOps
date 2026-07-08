import { useEffect, useRef, useState, useCallback } from "react";

interface WebSocketMessage {
  type: "thinking_start" | "response" | "problem_status" | "error";
  content?: string;
  intermediate_steps?: any[];
  status?: string;
  problem_id?: string;
  message?: string;
}

export function useSupportWebSocket(sessionId?: string) {
  const [isConnected, setIsConnected] = useState(false);
  const [response, setResponse] = useState("");
  const [isThinking, setIsThinking] = useState(false);
  const [steps, setSteps] = useState<any[]>([]);
  const wsRef = useRef<WebSocket | null>(null);

  const connect = useCallback(() => {
    const token = localStorage.getItem("token");
    const ws = new WebSocket(
      `ws://localhost:8000/ws/support/${sessionId || ""}?token=${token}`,
    );
    wsRef.current = ws;

    ws.onopen = () => {
      setIsConnected(true);
    };

    ws.onmessage = (event) => {
      const data: WebSocketMessage = JSON.parse(event.data);

      if (data.type === "thinking_start") {
        setIsThinking(true);
        setResponse("");
      } else if (data.type === "response") {
        setIsThinking(false);
        setResponse(data.content || "");
        setSteps(data.intermediate_steps || []);
      } else if (data.type === "problem_status") {
        console.log("Problem status:", data.status, data.problem_id);
      } else if (data.type === "error") {
        console.error("WebSocket error:", data.message);
        setIsThinking(false);
      }
    };

    ws.onclose = () => setIsConnected(false);
    ws.onerror = () => setIsConnected(false);
  }, [sessionId]);

  const sendMessage = useCallback(
    (message: string) => {
      if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "chat", message }));
      } else {
        connect();
        setTimeout(() => sendMessage(message), 500);
      }
    },
    [connect],
  );

  const submitProblem = useCallback((problemData: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: "problem", ...problemData }));
    }
  }, []);

  const disconnect = useCallback(() => {
    if (wsRef.current) wsRef.current.close();
  }, []);

  useEffect(() => {
    connect();
    return () => disconnect();
  }, [connect, disconnect]);

  return {
    isConnected,
    response,
    isThinking,
    steps,
    sendMessage,
    submitProblem,
  };
}
