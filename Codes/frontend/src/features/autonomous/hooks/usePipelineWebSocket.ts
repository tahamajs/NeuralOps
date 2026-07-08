import { useEffect, useState, useCallback, useRef } from "react";
import { PipelineRun, PipelineStep } from "../types";

interface WebSocketMessage {
  type: "status" | "step_update" | "log" | "progress" | "error" | "done";
  data: any;
}

export function usePipelineWebSocket(runId: string | null) {
  const [pipeline, setPipeline] = useState<PipelineRun | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [currentStep, setCurrentStep] = useState<PipelineStep | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const connect = useCallback(() => {
    if (!runId) return;
    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(
      `${protocol}://${window.location.host}/ws/research/${runId}/`,
    );
    wsRef.current = ws;
    ws.onopen = () => setIsConnected(true);
    ws.onmessage = (event) => {
      const msg: WebSocketMessage = JSON.parse(event.data);
      switch (msg.type) {
        case "status":
          setPipeline((prev) =>
            prev
              ? {
                  ...prev,
                  status: msg.data.status,
                  progress: msg.data.progress,
                }
              : msg.data,
          );
          break;
        case "step_update":
          setPipeline((prev) => {
            if (!prev) return prev;
            const updatedSteps = prev.steps.map((s) =>
              s.id === msg.data.id ? msg.data : s,
            );
            return {
              ...prev,
              steps: updatedSteps,
              current_step_id:
                msg.data.status === "running"
                  ? msg.data.id
                  : prev.current_step_id,
            };
          });
          setCurrentStep(msg.data);
          break;
        case "log":
          setLogs((prev) => [
            ...prev,
            `[${new Date().toLocaleTimeString()}] ${msg.data}`,
          ]);
          break;
        case "progress":
          setPipeline((prev) =>
            prev ? { ...prev, progress: msg.data } : prev,
          );
          break;
        case "error":
          setError(msg.data);
          break;
        case "done":
          setIsConnected(false);
          if (wsRef.current) wsRef.current.close();
          break;
      }
    };
    ws.onerror = () => setError("WebSocket connection error");
    ws.onclose = () => {
      setIsConnected(false);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = setTimeout(connect, 3000);
    };
  }, [runId]);

  useEffect(() => {
    connect();
    return () => {
      if (wsRef.current) wsRef.current.close();
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
    };
  }, [connect]);
  return { pipeline, logs, currentStep, isConnected, error };
}
