import { useEffect, useRef } from 'react'
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
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!terminalRef.current) return

    const term = new Terminal({
      cursorBlink: true,
      fontSize: 13,
      fontFamily: "'JetBrains Mono', 'Cascadia Code', monospace",
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
      scrollback: 5000,
    })

    const fitAddon = new FitAddon()
    const webLinksAddon = new WebLinksAddon()
    const searchAddon = new SearchAddon()

    term.loadAddon(fitAddon)
    term.loadAddon(webLinksAddon)
    term.loadAddon(searchAddon)

    term.open(terminalRef.current)
    fitAddon.fit()

    // Connect WebSocket to Django Channels terminal consumer
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const token = localStorage.getItem('token')
    const wsUrl = `${protocol}://${window.location.host}/ws/terminal/?token=${token}`
    const ws = new WebSocket(wsUrl)

    ws.onopen = () => {
      term.writeln('\x1b[1;36m╔══════════════════════════════════════╗\x1b[0m')
      term.writeln('\x1b[1;36m║   \x1b[1;37mNeuralOps Terminal v1.0\x1b[1;36m          ║\x1b[0m')
      term.writeln('\x1b[1;36m║   \x1b[0;32mConnected to cluster\x1b[1;36m              ║\x1b[0m')
      term.writeln('\x1b[1;36m╚══════════════════════════════════════╝\x1b[0m')
      term.writeln('')
    }

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'output') {
        term.write(data.data)
      } else if (data.type === 'error') {
        term.writeln(`\x1b[1;31m[ERROR]\x1b[0m ${data.data}`)
      }
    }

    ws.onerror = () => {
      term.writeln('\x1b[1;31m[Connection Error]\x1b[0m WebSocket connection failed.')
    }

    ws.onclose = () => {
      term.writeln('\x1b[1;33m[Disconnected]\x1b[0m Terminal session closed.')
    }

    // Send user input to backend
    term.onData((data) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: 'input', data }))
      }
    })

    terminalInstance.current = term
    wsRef.current = ws

    // Resize handler
    const handleResize = () => {
      fitAddon.fit()
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(
          JSON.stringify({
            type: 'resize',
            cols: term.cols,
            rows: term.rows,
          })
        )
      }
    }
    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      ws.close()
      term.dispose()
    }
  }, [])

  return (
    <div className="flex flex-col h-full">
      {/* Quick Command Bar */}
      <div className="flex items-center gap-2 px-3 py-1.5 bg-cyber-800 border-b border-white/5 text-[10px]">
        <span className="text-slate-500 font-bold uppercase tracking-wider">Quick:</span>
        {[
          { label: 'GPU', cmd: 'nvidia-smi\n' },
          { label: 'Jobs', cmd: 'python manage.py shell -c "from jobscheduler.models import Job; print(Job.objects.count())"\n' },
          { label: 'Logs', cmd: 'tail -f /var/log/neuralops/app.log\n' },
          { label: 'Clear', cmd: 'clear\n' },
        ].map((item) => (
          <button
            key={item.label}
            onClick={() => {
              const ws = wsRef.current
              if (ws?.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'input', data: item.cmd }))
              }
            }}
            className="px-2 py-0.5 rounded bg-white/5 hover:bg-neon-blue/20 text-slate-400 hover:text-neon-blue transition-colors"
          >
            {item.label}
          </button>
        ))}
      </div>
      {/* Terminal */}
      <div ref={terminalRef} className="flex-1 p-2" />
    </div>
  )
}