import { useEffect, useRef } from "react";
import { Terminal } from "@xterm/xterm";
import { FitAddon } from "@xterm/addon-fit";
import { WebLinksAddon } from "@xterm/addon-web-links";
import "@xterm/xterm/css/xterm.css";
import { integrationsAPI } from "../services/api";

export function TerminalEmulator({ sessionId }: { sessionId: string }) {
  const terminalRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const termRef = useRef<Terminal | null>(null);

  useEffect(() => {
    if (!terminalRef.current) return;
    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      theme: { background: "#0a0c10", foreground: "#e2e8f0" },
    });
    const fitAddon = new FitAddon();
    term.loadAddon(fitAddon);
    term.loadAddon(new WebLinksAddon());
    term.open(terminalRef.current);
    fitAddon.fit();
    termRef.current = term;
    integrationsAPI.getSandboxWebSocketUrl(sessionId).then((wsUrl: any) => {
      const ws = new WebSocket(wsUrl);
      ws.onopen = () =>
        term.write("\r\n\x1b[32mConnected to sandbox\x1b[0m\r\n$ ");
      ws.onmessage = (msg) => {
        const messageData = (msg as any).data || msg;
        term.write(
          typeof messageData === "string"
            ? messageData
            : JSON.stringify(messageData),
        );
      };
      wsRef.current = ws;
      term.onData((data) => ws.send(JSON.stringify({ type: "input", data })));
    });
    const resize = () => fitAddon.fit();
    window.addEventListener("resize", resize);
    return () => {
      wsRef.current?.close();
      term.dispose();
      window.removeEventListener("resize", resize);
    };
  }, [sessionId]);

  return (
    <div ref={terminalRef} style={{ height: "100%", minHeight: "400px" }} />
  );
}
