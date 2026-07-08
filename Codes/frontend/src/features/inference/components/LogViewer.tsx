import { useEffect, useRef } from "react";

export function LogViewer({ logs }: { logs: string[] }) {
  const containerRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (containerRef.current)
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
  }, [logs]);
  return (
    <TerminalCard className="p-2 h-64 overflow-y-auto font-mono text-xs bg-black/30">
      <div ref={containerRef}>
        {logs.map((line, i) => (
          <div key={i} className="text-slate-300 whitespace-pre-wrap">
            {line}
          </div>
        ))}
      </div>
    </TerminalCard>
  );
}
