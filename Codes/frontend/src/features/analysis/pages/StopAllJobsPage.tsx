import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Card, Button } from '../../../components/ui'
import { AlertTriangle } from 'lucide-react'

export default function StopAllJobsPage() {
  const [confirmed, setConfirmed] = useState(false)
  const stopAll = useMutation({
    mutationFn: () => api.post('/stop_all/'),
    onSuccess: () => alert('All jobs stopped successfully'),
  })

  return (
    <Card className="border-red-500/30 max-w-md mx-auto text-center">
      <AlertTriangle size={48} className="text-red-400 mx-auto mb-4" />
      <h1 className="text-xl font-bold text-white mb-2">Stop All Jobs</h1>
      <p className="text-sm text-slate-400 mb-4">This action cannot be undone. All running jobs will be terminated.</p>
      <label className="flex items-center justify-center gap-2 mb-4">
        <input type="checkbox" checked={confirmed} onChange={e => setConfirmed(e.target.checked)} />
        <span className="text-xs">I understand the consequences</span>
      </label>
      <Button variant="danger" disabled={!confirmed || stopAll.isPending} onClick={() => stopAll.mutate()}>
        {stopAll.isPending ? 'Stopping...' : 'Stop All Jobs'}
      </Button>
    </Card>
  )
}