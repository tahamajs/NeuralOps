import { useRef, useEffect } from "react";
import { TerminalCard } from "../../../components/ui/TerminalCard";

export function ExecutionLogs({ logs }: { logs: string[] }) {
  const logsEndRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);
  return (
    <TerminalCard className="h-full overflow-auto p-3 font-mono text-xs bg-black/50">
      <div className="space-y-1">
        {logs.map((log, idx) => (
          <div
            key={idx}
            className="text-slate-400 border-l-2 border-neon-blue pl-2"
          >
            {log}
          </div>
        ))}
        {logs.length === 0 && (
          <div className="text-slate-500 italic">Waiting for logs...</div>
        )}
        <div ref={logsEndRef} />
      </div>
    </TerminalCard>
  );
}
