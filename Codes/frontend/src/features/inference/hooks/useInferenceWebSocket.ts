import { useEffect, useState, useRef, useCallback } from "react";
import { useAuthStore } from "../../auth/store/authStore";

export function useInferenceWebSocket(runId: string | null) {
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const { token } = useAuthStore() as any;;

  const connect = useCallback(() => {
    if (!runId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/inference/${runId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "log") setLogs((prev) => [...prev, data.message]);
      else if (data.type === "status") setStatus(data.status);
    };
    ws.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 3000);
    };
  }, [runId, token]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);
  return { logs, status, isConnected };
}
