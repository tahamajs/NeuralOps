import { useEffect, useState, useRef, useCallback } from 'react'
import { useAuthStore } from '../../auth/store/authStore'

export function useWebSocketPrices(contractId: string | null) {
  const [prices, setPrices] = useState<{ current: number; bid: number; ask: number } | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const { token } = useAuthStore()

  useEffect(() => {
    if (!contractId) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/gpu-finance/${contractId}/?token=${token}`)
    wsRef.current = ws
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      setPrices(data)
    }
    ws.onclose = () => { setIsConnected(false); setTimeout(() => wsRef.current && new WebSocket(...), 3000) }
    return () => ws.close()
  }, [contractId, token])

  return { prices, isConnected }
}