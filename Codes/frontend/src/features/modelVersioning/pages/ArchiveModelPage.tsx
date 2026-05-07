import { useParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'

export default function ArchiveModelPage() {
  const { versionId } = useParams<{ versionId: string }>()
  const archiveMut = useMutation({
    mutationFn: () => api.post(`/api/model-versioning/versions/${versionId}/archive/`),
    onSuccess: () => alert('Archived to cold storage'),
  })
  const restoreMut = useMutation({
    mutationFn: () => api.post(`/api/model-versioning/versions/${versionId}/restore/`),
    onSuccess: () => alert('Restored from cold storage'),
  })

  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="text-center space-y-4 max-w-md w-full">
        <h1 className="text-2xl font-black text-white">COLD <span className="text-neon-blue">STORAGE</span></h1>
        <p className="text-slate-400 text-sm">Move model artifacts to cold storage to save space, or restore them.</p>
        <div className="flex gap-2 justify-center">
          <Button variant="secondary" onClick={() => archiveMut.mutate()} disabled={archiveMut.isPending}>
            Archive
          </Button>
          <Button variant="ghost" onClick={() => restoreMut.mutate()} disabled={restoreMut.isPending}>
            Restore
          </Button>
        </div>
      </Card>
    </div>
  )
}