import { useEffect, useState, useRef, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";

export function useInferenceWebSocket(runId: string | null) {
  const [logs, setLogs] = useState<string[]>([]);
  const [status, setStatus] = useState<string>("idle");
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!runId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = getAuthToken();
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/inference/${runId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "log") setLogs((prev) => [...prev, data.message]);
        else if (data.type === "status") setStatus(data.status);
      } catch (err) {
        console.error("Inference WS parse error", err);
      }
    };
    ws.onerror = () => setIsConnected(false);
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [runId]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { logs, status, isConnected };
}
