// src/hooks/useResearchStream.ts
import { useEffect, useState } from "react";

interface ResearchEvent {
  type: "node_status" | "custom_message" | "llm_token" | "final_report";
  nodeId?: string;
  status?: "pending" | "in_progress" | "completed" | "error";
  message?: string;
  token?: string;
  report?: string;
  progress?: { current: number; total: number };
}

export function useResearchStream(sessionId: string) {
  const [events, setEvents] = useState<ResearchEvent[]>([]);
  const [isActive, setIsActive] = useState(true);

  useEffect(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/research/${sessionId}`,
    );

    ws.onmessage = (msg) => {
      const data = JSON.parse(msg.data);
      setEvents((prev) => [...prev, data]);
      if (data.type === "final_report") setIsActive(false);
    };
    ws.onerror = () => console.error("Research stream error");
    return () => ws.close();
  }, [sessionId]);

  return { events, isActive };
}
