import { useEffect, useState, useRef, useCallback } from "react";
import { useAuthStore } from "../../auth/store/authStore";
import { DriftCheck } from "../types";

export function useDriftWebSocket(monitorId?: string) {
  const [driftChecks, setDriftChecks] = useState<DriftCheck[]>([]);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const { token } = useAuthStore();

  const connect = useCallback(() => {
    if (!monitorId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/drift/${monitorId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "drift_check") {
        setDriftChecks((prev) => [data.data, ...prev].slice(0, 50));
      }
    };
    ws.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 3000);
    };
  }, [monitorId, token]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);
  return { driftChecks, isConnected };
}
