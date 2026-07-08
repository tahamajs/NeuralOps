import { useEffect, useState, useCallback, useRef } from "react";
import { useAuthStore } from "../../auth/store/authStore";
import { Alert, SystemHealth } from "../types";

export function useHealthWebSocket(dashboardId?: string) {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const { token } = useAuthStore();

  const connect = useCallback(() => {
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/health/${dashboardId || "global"}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "alert") {
        setAlerts((prev) => [data.data, ...prev].slice(0, 100));
      } else if (data.type === "health_update") {
        setSystemHealth(data.data);
      }
    };
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [dashboardId, token]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);
  return { alerts, systemHealth, isConnected };
}
