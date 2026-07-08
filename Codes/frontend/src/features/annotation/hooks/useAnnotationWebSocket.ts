import { useEffect, useState, useRef, useCallback } from "react";
import { useAuthStore } from "../../auth/store/authStore";

export function useAnnotationWebSocket(projectId: string | null) {
  const [queueUpdates, setQueueUpdates] = useState<any[]>([]);
  const [preLabelProgress, setPreLabelProgress] = useState<number | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const { token } = useAuthStore() as any;

  const connect = useCallback(() => {
    if (!projectId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/annotation/${projectId}/?token=${token}`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === "queue_update")
        setQueueUpdates((prev) => [data.data, ...prev].slice(0, 50));
      else if (data.type === "pre_label_progress")
        setPreLabelProgress(data.progress);
    };
    ws.onclose = () => {
      setIsConnected(false);
      setTimeout(connect, 3000);
    };
  }, [projectId, token]);

  useEffect(() => {
    connect();
    return () => wsRef.current?.close();
  }, [connect]);
  return { queueUpdates, preLabelProgress, isConnected };
}
