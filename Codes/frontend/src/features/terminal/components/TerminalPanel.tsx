import { useEffect, useRef, useState, useCallback } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import { SearchAddon } from "@xterm/addon-search";
import "@xterm/xterm/css/xterm.css";

interface TerminalPanelProps {
  onClose?: () => void;
}

export default function TerminalPanel({ onClose }: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const terminalInstance = useRef<Terminal | null>(null);
  const fitAddonRef = useRef<FitAddon | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectAttemptRef = useRef(0);

  const [status, setStatus] = useState<
    "connected" | "connecting" | "disconnected" | "error"
  >("connecting");
  const [reconnectCountdown, setReconnectCountdown] = useState(0);

  const connectWebSocket = useCallback(() => {
    if (
      wsRef.current?.readyState === WebSocket.OPEN ||
      wsRef.current?.readyState === WebSocket.CONNECTING
    )
      return;

    const protocol = window.location.protocol === "https:" ? "wss" : "ws";
    const token = localStorage.getItem("token") || document.cookie.split("; ").find((row) => row.startsWith("authToken="))?.split("=")[1];
    const url = `${protocol}://${window.location.host}/ws/terminal/?token=${token}`;
    const ws = new WebSocket(url);
    wsRef.current = ws;
    setStatus("connecting");

    ws.onopen = () => {
      setStatus("connected");
      reconnectAttemptRef.current = 0;
      setReconnectCountdown(0);

      const term = terminalInstance.current;
      if (term) {
        term.clear();
        term.writeln(
          "\x1b[1;36m╔══════════════════════════════════════╗\x1b[0m",
        );
        term.writeln(
          "\x1b[1;36m║   \x1b[1;37mNeuralOps Terminal v2.0\x1b[1;36m          ║\x1b[0m",
        );
        term.writeln(
          "\x1b[1;36m║   \x1b[0;32mConnected to cluster\x1b[1;36m              ║\x1b[0m",
        );
        term.writeln(
          "\x1b[1;36m╚══════════════════════════════════════╝\x1b[0m",
        );
        term.writeln("");
      }
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        const term = terminalInstance.current;
        if (!term) return;
        if (data.type === "output") term.write(data.data);
        else if (data.type === "error")
          term.writeln(`\x1b[1;31m[ERROR]\x1b[0m ${data.data}`);
      } catch {
        terminalInstance.current?.write(event.data);
      }
    };

    ws.onerror = () => {
      setStatus("error");
      terminalInstance.current?.writeln("\x1b[1;31m[WebSocket Error]\x1b[0m");
    };

    ws.onclose = () => {
      setStatus("disconnected");
      terminalInstance.current?.writeln(
        "\x1b[1;33m[Disconnected]\x1b[0m Terminal session closed.",
      );

      const attempt = reconnectAttemptRef.current + 1;
      reconnectAttemptRef.current = attempt;

      if (attempt <= 3) {
        const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000);
        setReconnectCountdown(Math.floor(delay / 1000));
        reconnectTimerRef.current = setTimeout(() => {
          connectWebSocket();
        }, delay);
      } else {
        setStatus("error");
        terminalInstance.current?.writeln(
          "\x1b[1;31m[Error]\x1b[0m Max reconnects reached. Backend offline.",
        );
      }
    };
  }, []);

  useEffect(() => {
    if (!terminalRef.current) return;

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', monospace",
      theme: {
        background: "#0a0c10",
        foreground: "#e2e8f0",
        cursor: "#3b82f6",
        selectionBackground: "#3b82f640",
      },
      allowProposedApi: true,
      scrollback: 10000,
    });

    const fitAddon = new FitAddon();
    const webLinksAddon = new WebLinksAddon();
    const searchAddon = new SearchAddon();

    term.loadAddon(fitAddon);
    term.loadAddon(webLinksAddon);
    term.loadAddon(searchAddon);

    term.open(terminalRef.current);
    try {
      fitAddon.fit();
    } catch (e) {}
    fitAddonRef.current = fitAddon;

    term.onData((data) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: "input", data }));
      }
    });

    terminalInstance.current = term;
    connectWebSocket();

    const safeFit = (fitAddon: FitAddon, ws: WebSocket | null, term: Terminal) => {
      try {
        fitAddon.fit();
        if (ws?.readyState === WebSocket.OPEN && term.cols && term.rows) {
          ws.send(JSON.stringify({ type: "resize", cols: term.cols, rows: term.rows }));
        }
      } catch (err) {
        // Ignore xterm dimension errors when container is hidden
      }
    };

    const resizeObserver = new ResizeObserver(() => {
      safeFit(fitAddon, wsRef.current, term);
    });
    resizeObserver.observe(terminalRef.current);

    const handleResize = () => safeFit(fitAddon, wsRef.current, term);
    window.addEventListener("resize", handleResize);

    return () => {
      resizeObserver.disconnect();
      window.removeEventListener("resize", handleResize);
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current);
      wsRef.current?.close();
      term.dispose();
    };
  }, [connectWebSocket]);

  return (
    <div className="flex flex-col h-full bg-[#0a0c10]">
      {/* Quick command bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1f2e] border-b border-white/10 text-[10px] overflow-x-auto">
        <span className="text-slate-500 font-bold uppercase tracking-wider mr-1">
          Quick:
        </span>
        {[
          { label: "GPU", cmd: "nvidia-smi\n" },
          {
            label: "Jobs",
            cmd: 'python manage.py shell -c "from jobscheduler.models import Job; print(Job.objects.count())"\n',
          },
          { label: "Logs", cmd: "tail -f /var/log/neuralops/app.log\n" },
          { label: "Clear", cmd: "clear\n" },
        ].map((item) => (
          <button
            key={item.label}
            onClick={() => {
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(
                  JSON.stringify({ type: "input", data: item.cmd }),
                );
              }
            }}
            className="px-2 py-0.5 rounded bg-white/5 hover:bg-neon-blue/20 text-slate-400 hover:text-neon-blue transition-colors whitespace-nowrap"
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* Status bar */}
      <div className="flex items-center justify-between px-3 py-0.5 bg-[#1a1f2e] border-b border-white/10 text-[10px]">
        <span className="text-slate-500 font-bold uppercase tracking-wider">
          TERMINAL • PID $$
        </span>
        <span
          className={`font-bold uppercase tracking-wider ${
            status === "connected"
              ? "text-green-400"
              : status === "connecting"
                ? "text-yellow-400"
                : "text-red-400"
          }`}
        >
          {status === "connected" && "● CONNECTED"}
          {status === "connecting" && "◌ CONNECTING"}
          {status === "disconnected" &&
            reconnectCountdown > 0 &&
            `⟳ RETRY IN ${reconnectCountdown}s`}
          {status === "error" && "⚠ ERROR"}
          {status === "disconnected" && reconnectCountdown === 0 && "○ OFFLINE"}
        </span>
      </div>

      {/* xterm.js container */}
      <div ref={terminalRef} className="flex-1 p-1" />
    </div>
  );
}
