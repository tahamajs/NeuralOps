import { useEffect, useState, useRef } from 'react'
import { useAuthStore } from '../../auth/store/authStore'
import { OrderBookDepth } from '../types'

export function useOrderBookWebSocket(listingId: string | null) {
  const [liveDepth, setLiveDepth] = useState<OrderBookDepth | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null)
  const { token } = useAuthStore()

  useEffect(() => {
    if (!listingId) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const wsUrl = `${protocol}://${window.location.host}/ws/marketplace/depth/${listingId}/?token=${token}`
    let isUnmounted = false

    const connect = () => {
      if (isUnmounted) return
      const ws = new WebSocket(wsUrl)
      wsRef.current = ws
      ws.onopen = () => setIsConnected(true)
      ws.onmessage = event => setLiveDepth(JSON.parse(event.data))
      ws.onclose = () => {
        setIsConnected(false)
        if (!isUnmounted) {
          reconnectTimerRef.current = setTimeout(connect, 3000)
        }
      }
    }

    connect()

    return () => {
      isUnmounted = true
      if (reconnectTimerRef.current) clearTimeout(reconnectTimerRef.current)
      wsRef.current?.close()
    }
  }, [listingId, token])

  return { liveDepth, isConnected }
}