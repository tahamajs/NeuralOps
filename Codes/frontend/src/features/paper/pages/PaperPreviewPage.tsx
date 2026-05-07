import { useParams } from 'react-router-dom'
import { Card } from '../../../components/ui/Card'

export default function PaperPreviewPage() {
  const { runId } = useParams<{ runId: string }>()
  return (
    <div className="space-y-6 animate-fade-in">
      <h1 className="text-2xl font-black text-white">PAPER <span className="text-neon-blue">PREVIEW</span></h1>
      <Card className="p-0 overflow-hidden">
        <iframe
          src={`/api/runs/${runId}/paper_preview/`}
          className="w-full h-[800px] border-0"
          title="Paper Preview"
        />
      </Card>
    </div>
  )
}