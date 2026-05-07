import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { useLiveStore } from '../../../stores/experimentStore'
import { useWebSocket } from '../../../hooks/useWebSocket'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import ProgressBar from '../../../components/ProgressBar'  // we'll create a re-usable one
import LiveLossPlot from '../../../components/LiveLossPlot'
import LiveLogViewer from '../../../components/LiveLogViewer'
import ConfigSummary from '../../../components/ConfigSummary'
import { Experiment } from '../../../types/experiment'

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>()
  useWebSocket(id)

  const { data: experiment, isLoading } = useQuery<Experiment>({
    queryKey: ['experiment', id],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.DETAIL(id!))).data,
    enabled: !!id,
  })

  const liveStore = useLiveStore()

  if (isLoading || !experiment) return <Spinner />

  // Merge live status/metrics
  const status = liveStore.status || experiment.status
  const progress = liveStore.progress || { current: experiment.current_step, total: experiment.total_steps, eta: 0 }
  const logs = liveStore.logs.length > 0 ? liveStore.logs : experiment.logs || []
  const metrics = liveStore.metrics || { loss: [], accuracy: [] }

  return (
    <div className="space-y-6 animate-fade-in">
      <div className="flex flex-wrap justify-between items-center gap-4">
        <h1 className="text-2xl font-black text-white tracking-tighter">
          {experiment.name || <span className="text-slate-400">Unnamed Session</span>}
        </h1>
        <Badge status={status} />
      </div>

      <ProgressBar progress={progress} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <LiveLossPlot metrics={metrics} />
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Configuration</h3>
          <ConfigSummary config={experiment.config} />
        </Card>
      </div>

      <Card>
        <h3 className="text-sm font-semibold text-white mb-3">Live Logs</h3>
        <LiveLogViewer logs={logs} />
      </Card>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Artifacts</h3>
          <ArtifactList experimentId={id!} />
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Model</h3>
          <a
            href={ENDPOINTS.EXPERIMENTS.MODEL(id!)}
            className="inline-block bg-neon-blue/80 hover:bg-neon-blue text-white text-sm font-bold py-2 px-4 rounded-lg transition-colors"
            download
          >
            Download Model (ZIP)
          </a>
        </Card>
      </div>
    </div>
  )
}