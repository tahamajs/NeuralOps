import { useEffect, useState, useRef } from 'react'
import { useAuthStore } from '../../auth/store/authStore'

export function useSocialWebSocket() {
  const [newAlert, setNewAlert] = useState<any>(null)
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const { token } = useAuthStore()

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/social/alerts/?token=${token}`)
    wsRef.current = ws
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => setNewAlert(JSON.parse(event.data))
    ws.onclose = () => { setIsConnected(false); setTimeout(() => wsRef.current && new WebSocket(...), 3000) }
    return () => ws.close()
  }, [token])

  return { newAlert, isConnected }
}