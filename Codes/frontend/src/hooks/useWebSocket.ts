import { useEffect, useRef } from 'react'
import { useLiveStore } from '../store/experimentStore'

export function useWebSocket(experimentId: string | undefined) {
  const store = useLiveStore()
  const wsRef = useRef<WebSocket | null>(null)

  useEffect(() => {
    if (!experimentId) return
    store.reset()

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const url = `${protocol}://${window.location.host}/ws/experiments/${experimentId}/`
    const socket = new WebSocket(url)
    wsRef.current = socket

    socket.onmessage = (event) => {
      const msg = JSON.parse(event.data)
      switch (msg.type) {
        case 'status':
          store.updateStatus(msg.status)
          break
        case 'log':
          store.addLog(msg.data)
          break
        case 'metrics':
          store.updateMetrics(msg.data)
          break
        case 'progress':
          store.updateProgress(msg)
          break
        case 'gpu':
          store.updateGpu(msg)
          break
      }
    }

    return () => {
      socket.close()
    }
  }, [experimentId])

  return wsRef
}