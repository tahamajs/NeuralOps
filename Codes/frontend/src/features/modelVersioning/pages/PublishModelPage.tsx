import { useParams } from 'react-router-dom'
import { useMutation } from '@tanstack/react-query'
import api from '../../../api/client'
import { Button } from '../../../components/ui/Button'
import { Card } from '../../../components/ui/Card'

export default function PublishModelPage() {
  const { versionId } = useParams<{ versionId: string }>()
  const mutation = useMutation({
    mutationFn: async () => api.post(`/api/model-versioning/versions/${versionId}/publish/`),
    onSuccess: () => alert('Published to HuggingFace Hub'),
  })
  return (
    <div className="min-h-[80vh] flex items-center justify-center">
      <Card className="text-center space-y-4 max-w-md w-full">
        <h1 className="text-2xl font-black text-white">PUBLISH <span className="text-neon-blue">MODEL</span></h1>
        <p className="text-slate-400 text-sm">Publish this model version to the HuggingFace Hub.</p>
        <Button onClick={() => mutation.mutate()} disabled={mutation.isPending}>
          {mutation.isPending ? 'Publishing...' : 'Publish Now'}
        </Button>
      </Card>
    </div>
  )
}