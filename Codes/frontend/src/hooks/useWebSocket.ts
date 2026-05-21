// src/hooks/useWebSocket.ts
import { useEffect, useRef, useState } from 'react'
import { useLiveStore } from '../stores/experimentStore'

export type WSStatus = 'connecting' | 'open' | 'closed' | 'error'

export function useWebSocket(experimentId: string | undefined) {
  const store = useLiveStore()
  const [wsStatus, setWsStatus] = useState<WSStatus>('closed')
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  useEffect(() => {
    if (!experimentId) return
    store.reset()

    const connect = () => {
      setWsStatus('connecting')
      const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
      const wsUrl = import.meta.env.VITE_WS_URL || `${protocol}://${window.location.host}`
      const url = `${wsUrl}/ws/experiments/${experimentId}/`
      const socket = new WebSocket(url)
      wsRef.current = socket

      socket.onopen = () => {
        setWsStatus('open')
        store.addLog('Connected to live stream')
      }

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data)
          switch (msg.type) {
            case 'status': store.updateStatus(msg.status); break
            case 'log': store.addLog(msg.data); break
            case 'metrics': store.updateMetrics(msg.data); break
            case 'progress': store.updateProgress(msg); break
            case 'gpu': store.updateGpu?.(msg); break
            case 'final': store.updateMetrics(msg.data); store.addLog('Job completed'); break
          }
        } catch (err) {
          console.error('WS parse error', err)
        }
      }

      socket.onerror = () => {
        setWsStatus('error')
        store.addLog('WebSocket error')
      }

      socket.onclose = () => {
        setWsStatus('closed')
        store.addLog('Connection closed. Reconnecting in 3s...')
        if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
        reconnectTimerRef.current = setTimeout(connect, 3000)
      }
    }

    connect()

    return () => {
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [experimentId])

  return { wsStatus, ws: wsRef }
}