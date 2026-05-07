import React, { createContext, useContext, useEffect, useState } from 'react'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { HardwareMetrics } from '../../../types/hardware'

interface HardwareContextType {
  metrics: HardwareMetrics
  isLoading: boolean
  error: string | null
}

const HardwareContext = createContext<HardwareContextType | undefined>(undefined)

export const HardwareProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [metrics, setMetrics] = useState<HardwareMetrics>({ cpu: 0, memory: 0, gpu: 0 })
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    const fetch = async () => {
      setIsLoading(true)
      try {
        const [sysRes, gpuRes] = await Promise.all([
          api.get(ENDPOINTS.SYSTEM.STATUS),
          api.get(ENDPOINTS.SYSTEM.GPU),
        ])
        setMetrics({
          cpu: sysRes.data.cpu?.percent || 0,
          memory: sysRes.data.memory?.percent || 0,
          gpu: gpuRes.data.devices?.[0]?.utilization_percent || 0,
        })
        setError(null)
      } catch (err) {
        setError('Failed to fetch hardware metrics')
      } finally {
        setIsLoading(false)
      }
    }

    fetch()
    const interval = setInterval(fetch, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <HardwareContext.Provider value={{ metrics, isLoading, error }}>
      {children}
    </HardwareContext.Provider>
  )
}

export const useHardware = () => {
  const context = useContext(HardwareContext)
  if (!context) throw new Error('useHardware must be used within HardwareProvider')
  return context
}