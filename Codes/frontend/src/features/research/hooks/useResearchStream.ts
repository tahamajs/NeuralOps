import { useState, useEffect, useRef } from "react";

export interface StreamEvent {
  type: "llm_token" | "status" | "final_report" | "error";
  token?: string;
  nodeId?: string;
  message?: string;
  progress?: { current: number; total: number };
  report?: string;
}

export function useResearchStream(sessionId: string) {
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [isActive, setIsActive] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!sessionId) {
      setIsActive(false);
      setEvents([]);
      return;
    }

    setIsActive(true);
    setEvents([
      {
        type: "status",
        nodeId: "system",
        message: "Initializing Deep Research Agent session...",
        progress: { current: 1, total: 10 },
      },
    ]);

    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host || "localhost:8000";
    const wsUrl = `${protocol}//${host}/ws/agent/${sessionId}/`;

    try {
      const socket = new WebSocket(wsUrl);
      socketRef.current = socket;

      socket.onopen = () => {
        setEvents((prev) => [
          ...prev,
          {
            type: "status",
            nodeId: "websocket",
            message: "Connected to research server. Initiating agent executor...",
            progress: { current: 2, total: 10 },
          },
        ]);
        socket.send(
          JSON.stringify({
            type: "message",
            text: "Generate research report.",
          }),
        );
      };

      socket.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "token") {
            setEvents((prev) => [...prev, { type: "llm_token", token: data.text }]);
          } else if (data.type === "status") {
            setEvents((prev) => [
              ...prev,
              {
                type: "status",
                nodeId: data.nodeId,
                message: data.message,
                progress: data.progress,
              },
            ]);
          } else if (data.type === "done") {
            setEvents((prev) => [
              ...prev,
              {
                type: "status",
                nodeId: "system",
                message: "Deep Research finished.",
                progress: { current: 10, total: 10 },
              },
              {
                type: "final_report",
                report: data.report || "# Research Report\n\nCompleted.",
              },
            ]);
            setIsActive(false);
          }
        } catch (err) {
          console.error("Error parsing socket JSON:", err);
        }
      };

      socket.onerror = (err) => {
        setEvents((prev) => [
            ...prev,
            { type: "error", message: "WebSocket connection failed." }
        ]);
        setIsActive(false);
      };

      socket.onclose = () => {
        setIsActive(false);
      };
    } catch (e) {
      setEvents((prev) => [
        ...prev,
        { type: "error", message: "Failed to initialize WebSocket connection." }
      ]);
      setIsActive(false);
    }

    return () => {
      if (socketRef.current) {
        socketRef.current.close();
      }
    };
  }, [sessionId]);

  return { events, isActive };
}
