import { useEffect, useRef, useState, useCallback } from 'react'
import { Terminal } from '@xterm/xterm'
import { FitAddon } from '@xterm/addon-fit'
import { WebLinksAddon } from '@xterm/addon-web-links'
import { SearchAddon } from '@xterm/addon-search'
import '@xterm/xterm/css/xterm.css'

interface TerminalPanelProps {
  onClose?: () => void
}

export default function TerminalPanel({ onClose }: TerminalPanelProps) {
  const terminalRef = useRef<HTMLDivElement>(null)
  const terminalInstance = useRef<Terminal | null>(null)
  const fitAddonRef = useRef<FitAddon | null>(null)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const reconnectAttemptRef = useRef(0)

  const [status, setStatus] = useState<'connected' | 'connecting' | 'disconnected' | 'error'>('connecting')
  const [reconnectCountdown, setReconnectCountdown] = useState(0)

  const connectWebSocket = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN || wsRef.current?.readyState === WebSocket.CONNECTING) return

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const token = localStorage.getItem('token')
    const url = `${protocol}://${window.location.host}/ws/terminal/?token=${token}`
    const ws = new WebSocket(url)
    wsRef.current = ws
    setStatus('connecting')

    ws.onopen = () => {
      setStatus('connected')
      reconnectAttemptRef.current = 0
      setReconnectCountdown(0)

      const term = terminalInstance.current
      if (term) {
        term.clear()
        term.writeln('\x1b[1;36m╔══════════════════════════════════════╗\x1b[0m')
        term.writeln('\x1b[1;36m║   \x1b[1;37mNeuralOps Terminal v2.0\x1b[1;36m          ║\x1b[0m')
        term.writeln('\x1b[1;36m║   \x1b[0;32mConnected to cluster\x1b[1;36m              ║\x1b[0m')
        term.writeln('\x1b[1;36m╚══════════════════════════════════════╝\x1b[0m')
        term.writeln('')
        term.writeln('Type \x1b[1;33mhelp\x1b[0m for available commands.')
      }
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        const term = terminalInstance.current
        if (!term) return

        if (data.type === 'output') {
          term.write(data.data)
        } else if (data.type === 'error') {
          term.writeln(`\x1b[1;31m[ERROR]\x1b[0m ${data.data}`)
        }
      } catch {
        // Non‑JSON message, treat as raw text
        terminalInstance.current?.write(event.data)
      }
    }

    ws.onerror = () => {
      setStatus('error')
      terminalInstance.current?.writeln('\x1b[1;31m[WebSocket Error]\x1b[0m')
    }

    ws.onclose = () => {
      setStatus('disconnected')
      terminalInstance.current?.writeln('\x1b[1;33m[Disconnected]\x1b[0m Terminal session closed.')

      const attempt = reconnectAttemptRef.current + 1
      reconnectAttemptRef.current = attempt
      const delay = Math.min(1000 * Math.pow(2, attempt - 1), 30000)
      setReconnectCountdown(Math.floor(delay / 1000))

      reconnectTimerRef.current = setTimeout(() => {
        connectWebSocket()
      }, delay)
    }
  }, [])

  useEffect(() => {
    if (!terminalRef.current) return

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', 'Fira Code', monospace",
      theme: {
        background: '#0a0c10',
        foreground: '#e2e8f0',
        cursor: '#3b82f6',
        selectionBackground: '#3b82f640',
        black: '#1e2330',
        red: '#ef4444',
        green: '#10b981',
        yellow: '#f59e0b',
        blue: '#3b82f6',
        magenta: '#8b5cf6',
        cyan: '#06b6d4',
        white: '#e2e8f0',
        brightBlack: '#3f4a60',
        brightRed: '#f87171',
        brightGreen: '#34d399',
        brightYellow: '#fbbf24',
        brightBlue: '#60a5fa',
        brightMagenta: '#a78bfa',
        brightCyan: '#22d3ee',
        brightWhite: '#ffffff',
      },
      allowProposedApi: true,
      scrollback: 10000,
      lineHeight: 1.2,
    })

    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()
    const searchAddon = new SearchAddon()

    term.loadAddon(fitAddon)
    term.loadAddon(webLinksAddon)
    term.loadAddon(searchAddon)

    term.open(terminalRef.current)
    fitAddon.fit()
    fitAddonRef.current = fitAddon

    // Send input to WebSocket
    term.onData((data) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify({ type: 'input', data }))
      }
    })

    // Ctrl+Shift+C (copy) and Ctrl+Shift+V (paste)
    term.attachCustomKeyEventHandler((e) => {
      if (e.ctrlKey && e.shiftKey && e.key === 'C') {
        const selection = term.getSelection()
        if (selection) navigator.clipboard.writeText(selection)
        return false
      }
      if (e.ctrlKey && e.shiftKey && e.key === 'V') {
        navigator.clipboard.readText().then(text => {
          if (wsRef.current?.readyState === WebSocket.OPEN) {
            wsRef.current.send(JSON.stringify({ type: 'input', data: text }))
          }
        })
        return false
      }
      return true
    })

    terminalInstance.current = term
    connectWebSocket()

    // Resize observer for container size changes (e.g., floating window resize)
    const resizeObserver = new ResizeObserver(() => {
      fitAddon.fit()
      const ws = wsRef.current
      if (ws?.readyState === WebSocket.OPEN && term) {
        ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }))
      }
    })
    resizeObserver.observe(terminalRef.current)

    // Also listen to window resize (for good measure)
    const handleWindowResize = () => {
      fitAddon.fit()
      const ws = wsRef.current
      if (ws?.readyState === WebSocket.OPEN && term) {
        ws.send(JSON.stringify({ type: 'resize', cols: term.cols, rows: term.rows }))
      }
    }
    window.addEventListener('resize', handleWindowResize)

    return () => {
      resizeObserver.disconnect()
      window.removeEventListener('resize', handleWindowResize)
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
      term.dispose()
    }
  }, [connectWebSocket])

  // Handle right‑click context menu for copy/paste
  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault()
    const term = terminalInstance.current
    if (!term) return
    const selection = term.getSelection()
    if (selection) {
      navigator.clipboard.writeText(selection)
      // Optionally show a small notification
    }
  }

  return (
    <div className="flex flex-col h-full bg-[#0a0c10]" onContextMenu={handleContextMenu}>
      {/* Quick Command Toolbar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-[#1a1f2e] border-b border-white/10 text-[10px] overflow-x-auto">
        <span className="text-slate-500 font-bold uppercase tracking-wider mr-1">Quick:</span>
        {[
          { label: 'GPU', cmd: 'nvidia-smi\n' },
          { label: 'Jobs', cmd: 'python manage.py shell -c "from jobscheduler.models import Job; print(Job.objects.count())"\n' },
          { label: 'Logs', cmd: 'tail -f /var/log/neuralops/app.log\n' },
          { label: 'Clear', cmd: 'clear\n' },
          { label: 'Help', cmd: 'help\n' },
        ].map(item => (
          <button
            key={item.label}
            onClick={() => {
              if (wsRef.current?.readyState === WebSocket.OPEN) {
                wsRef.current.send(JSON.stringify({ type: 'input', data: item.cmd }))
              }
            }}
            className="px-2 py-0.5 rounded bg-white/5 hover:bg-neon-blue/20 text-slate-400 hover:text-neon-blue transition-colors whitespace-nowrap"
          >
            {item.label}
          </button>
        ))}
      </div>

      {/* Status Bar */}
      <div className="flex items-center justify-between px-3 py-0.5 bg-[#1a1f2e] border-b border-white/10 text-[10px]">
        <span className="text-slate-500 font-bold uppercase tracking-wider">
          TERMINAL • PID $$
        </span>
        <span
          className={`font-bold uppercase tracking-wider ${
            status === 'connected' ? 'text-green-400' : status === 'connecting' ? 'text-yellow-400' : 'text-red-400'
          }`}
        >
          {status === 'connected' && '● CONNECTED'}
          {status === 'connecting' && '◌ CONNECTING'}
          {status === 'disconnected' && reconnectCountdown > 0 && `⟳ RETRY IN ${reconnectCountdown}s`}
          {status === 'error' && '⚠ ERROR'}
          {status === 'disconnected' && reconnectCountdown === 0 && '○ OFFLINE'}
        </span>
      </div>

      {/* xterm.js container */}
      <div ref={terminalRef} className="flex-1 p-1" />

      {/* Optional: Close button if needed (already in window title bar) */}
      {onClose && (
        <button
          onClick={onClose}
          className="absolute top-2 right-2 text-slate-400 hover:text-white text-xs"
          style={{ display: 'none' }} // hidden because window title bar has close button
        />
      )}
    </div>
  )
}