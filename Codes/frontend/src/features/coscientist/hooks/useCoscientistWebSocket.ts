// src/features/coscientist/hooks/useCoscientistWebSocket.ts
import { useEffect, useState, useCallback } from 'react'
import { Hypothesis, TournamentMatch, Agent } from '../types'

interface WebSocketMessage {
  type: 'agent_update' | 'new_hypothesis' | 'tournament_update' | 'session_complete' | 'error'
  data: any
}

export function useCoscientistWebSocket(sessionId: string | null) {
  const [agents, setAgents] = useState<Agent[]>([])
  const [hypotheses, setHypotheses] = useState<Hypothesis[]>([])
  const [matches, setMatches] = useState<TournamentMatch[]>([])
  const [isConnected, setIsConnected] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!sessionId) return

    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws'
    const ws = new WebSocket(`${protocol}://${window.location.host}/ws/coscientist/${sessionId}/`)
    ws.onopen = () => setIsConnected(true)
    ws.onmessage = (event) => {
      const msg: WebSocketMessage = JSON.parse(event.data)
      switch (msg.type) {
        case 'agent_update':
          setAgents(msg.data)
          break
        case 'new_hypothesis':
          setHypotheses(prev => [...prev, msg.data])
          break
        case 'tournament_update':
          setMatches(msg.data)
          break
        case 'session_complete':
          setIsConnected(false)
          break
        case 'error':
          setError(msg.data)
          break
      }
    }
    ws.onerror = () => setError('WebSocket connection error')
    ws.onclose = () => setIsConnected(false)

    return () => ws.close()
  }, [sessionId])

  return { agents, hypotheses, matches, isConnected, error }
}