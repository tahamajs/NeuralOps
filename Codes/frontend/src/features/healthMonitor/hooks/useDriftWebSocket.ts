import { useEffect, useState, useRef, useCallback } from "react";
import { getAuthToken } from "../../../utils/authUtils";
import { DriftCheck } from "../types";

export function useDriftWebSocket(monitorId?: string) {
  const [driftChecks, setDriftChecks] = useState<DriftCheck[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!monitorId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = getAuthToken();
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/drift/${monitorId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        if (data.type === "drift_check") {
          setDriftChecks((prev) => [data.data, ...prev].slice(0, 50));
        }
      } catch (err) {
        console.error("Drift WebSocket parse error", err);
      }
    };
    ws.onerror = () => setIsConnected(false);
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [monitorId]);

  useEffect(() => {
    connect();
    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
    };
  }, [connect]);

  return { driftChecks, isConnected };
}
