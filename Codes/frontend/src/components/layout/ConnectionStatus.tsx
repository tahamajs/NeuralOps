import { useEffect, useState } from 'react'
import api from '../../api/client'

export default function ConnectionStatus() {
  const [online, setOnline] = useState(navigator.onLine)
  const [apiOnline, setApiOnline] = useState(true)

  useEffect(() => {
    const handleOnline = () => setOnline(true)
    const handleOffline = () => setOnline(false)
    window.addEventListener('online', handleOnline)
    window.addEventListener('offline', handleOffline)
    return () => {
      window.removeEventListener('online', handleOnline)
      window.removeEventListener('offline', handleOffline)
    }
  }, [])

  useEffect(() => {
    const checkApi = async () => {
      try {
        await api.get('/health/', { timeout: 3000 })
        setApiOnline(true)
      } catch {
        setApiOnline(false)
      }
    }
    checkApi()
    const interval = setInterval(checkApi, 10000)
    return () => clearInterval(interval)
  }, [])

  const isConnected = online && apiOnline
  if (isConnected) return null

  return (
    <div className={`sticky top-0 z-50 w-full text-center py-2 text-xs font-bold uppercase tracking-wider transition-all ${
      !online ? 'bg-red-600 text-white' : 'bg-yellow-600 text-black'
    }`}>
      {!online
        ? '⚠️ No internet connection – some features are unavailable'
        : '⚠️ Backend is unreachable – retrying automatically...'}
    </div>
  )
}