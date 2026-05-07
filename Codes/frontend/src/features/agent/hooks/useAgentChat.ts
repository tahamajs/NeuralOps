import { useEffect, useRef, useState, useCallback } from 'react'
import api from '../../../api/client'
import { ChatMessage } from '../types'

export function useAgentChat(sessionId?: string) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [loading, setLoading] = useState(false)
  const [sending, setSending] = useState(false)
  const wsRef = useRef<WebSocket | null>(null)

  // Load initial messages
  useEffect(() => {
    if (!sessionId) return
    api.get(`/api/agents/chats/${sessionId}/messages/`)
      .then(res => setMessages(res.data))
      .catch(console.error)
  }, [sessionId])

  // WebSocket connection
  useEffect(() => {
    if (!sessionId) return
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const host = import.meta.env.VITE_WS_URL || `${protocol}://${window.location.host}`
    const ws = new WebSocket(`${host}/ws/agent/${sessionId}/`)

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data)
      if (data.type === 'messages_sync') {
        setMessages(data.messages)
        setLoading(false)
        setSending(false)
      }
    }
    ws.onerror = () => { setSending(false); setLoading(false) }
    ws.onclose = () => { setSending(false) }

    wsRef.current = ws
    return () => ws.close()
  }, [sessionId])

  const sendMessage = useCallback((text: string) => {
    if (!text.trim() || !sessionId) return
    setSending(true)
    // Add optimistic user message
    const optimisticMsg: ChatMessage = {
      id: Date.now().toString(),
      role: 'user',
      content: text,
      created_at: new Date().toISOString(),
    }
    setMessages(prev => [...prev, optimisticMsg])

    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'message', text }))
    } else {
      // REST fallback
      setLoading(true)
      api.post(`/api/agents/chats/${sessionId}/send/`, { text })
        .then(res => {
          setMessages(res.data.messages)
          setLoading(false)
          setSending(false)
        })
        .catch(() => { setLoading(false); setSending(false) })
    }
  }, [sessionId])

  return { messages, loading, sending, sendMessage }
}