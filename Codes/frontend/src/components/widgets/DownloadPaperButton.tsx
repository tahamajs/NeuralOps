import { Button } from '../ui/Button'

export const DownloadModelButton = ({ experimentId }: { experimentId: string }) => (
  <div>
    <p className="text-xs text-slate-400 mb-2">Download the trained model (ZIP archive).</p>
    <a href={`/api/experiments/${experimentId}/model/`} download>
      <Button variant="secondary" size="sm">Download Model</Button>
    </a>
  </div>
)