import { useEffect, useState, useRef, useCallback } from 'react'
import { useAuthStore } from '../../auth/store/authStore'
import { OrderBookDepth } from '../types'

export function useOrderBookWebSocket(listingId: string | null) {
  const [liveDepth, setLiveDepth] = useState<OrderBookDepth | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const { token } = useAuthStore()

  useEffect(() => {
    if (!listingId) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/marketplace/depth/${listingId}/?token=${token}`)
    wsRef.current = ws
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => setLiveDepth(JSON.parse(event.data))
    ws.onclose = () => { setIsConnected(false); setTimeout(() => wsRef.current && new WebSocket(...), 3000) }
    return () => ws.close()
  }, [listingId, token])

  return { liveDepth, isConnected }
}