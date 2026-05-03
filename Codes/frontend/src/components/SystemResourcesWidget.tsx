import { useEffect, useState } from 'react'
import api from '../api/client'

export default function SystemResourcesWidget() {
  const [sys, setSys] = useState({ cpu: 0, memory: 0 })
  const [gpu, setGpu] = useState({ cuda_available: false, memory_used: 0, memory_total: 0 })

  useEffect(() => {
    const fetch = async () => {
      try {
        const s = await api.get('/system/status/')
        setSys(s.data)
        const g = await api.get('/system/gpu/')
        setGpu(g.data)
      } catch {}
    }
    fetch()
    const interval = setInterval(fetch, 5000)
    return () => clearInterval(interval)
  }, [])

  return (
    <div className="bg-white rounded-lg shadow p-4">
      <h2 className="text-lg font-semibold mb-2">System Resources</h2>
      <div className="flex gap-4 text-sm">
        <div>CPU: {sys.cpu}%</div>
        <div>RAM: {sys.memory}%</div>
        {gpu.cuda_available && (
          <div>GPU: {gpu.memory_used?.toFixed(1)} / {gpu.memory_total?.toFixed(1)} GB</div>
        )}
      </div>
    </div>
  )
}