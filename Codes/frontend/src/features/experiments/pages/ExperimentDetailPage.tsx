import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useWebSocket } from '../../../hooks/useWebSocket'
import { useLiveStore } from '../../../stores/experimentStore'
import api from '../../../api/client'
import { ENDPOINTS } from '../../../api/endpoints'
import { Card } from '../../../components/ui/Card'
import { Badge } from '../../../components/ui/Badge'
import { Spinner } from '../../../components/ui/Spinner'
import { LiveLossPlot } from '../../../components/widgets/LiveLossPlot'
import { LiveLogViewer } from '../../../components/widgets/LiveLogViewer'
import { ProgressBar } from '../../../components/widgets/ProgressBar'
import { ConfigSummary } from '../../../components/widgets/ConfigSummary'
import { ArtifactList } from '../../../components/widgets/ArtifactList'
import { DownloadModelButton } from '../../../components/widgets/DownloadModelButton'
import { DownloadPaperButton } from '../../../components/widgets/DownloadPaperButton'
import { Experiment } from '../../../types/experiment'

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>()
  useWebSocket(id)

  const { data: experiment, isLoading } = useQuery<Experiment>({
    queryKey: ['experiment', id],
    queryFn: async () => (await api.get(ENDPOINTS.EXPERIMENTS.DETAIL(id!))).data,
    enabled: !!id,
    refetchInterval: 10000,
  })

  const live = useLiveStore()

  if (isLoading || !experiment) return <Spinner className="py-16" />

  // Merge live data from WebSocket with initial experiment data
  const status = live.status || experiment.status
  const progress = live.progress || { current: experiment.current_step, total: experiment.total_steps, eta: 0 }
  const logs = live.logs.length > 0 ? live.logs : experiment.logs || []
  const metrics = live.metrics.loss.length > 0 ? live.metrics : experiment.metrics || { loss: [], accuracy: [] }

  return (
    <div className="space-y-6 animate-fade-in">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-3xl font-black text-white tracking-tighter">
            {experiment.name || 'Unnamed Session'}
          </h1>
          <p className="text-xs text-slate-400 font-mono mt-1">{experiment.id}</p>
        </div>
        <Badge status={status} />
      </div>

      {/* Progress Bar */}
      <Card>
        <ProgressBar progress={progress} />
      </Card>

      {/* Metrics & Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-neon-blue rounded-full" />
            Training Loss
          </h3>
          <LiveLossPlot metrics={metrics} />
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
            <span className="w-1 h-4 bg-purple-500 rounded-full" />
            Configuration
          </h3>
          <ConfigSummary config={experiment.config} />
        </Card>
      </div>

      {/* Live Logs */}
      <Card>
        <h3 className="text-sm font-semibold text-white mb-3 flex items-center gap-2">
          <span className="w-1 h-4 bg-green-500 rounded-full" />
          Live Logs
        </h3>
        <LiveLogViewer logs={logs} />
      </Card>

      {/* Artifacts, Model, Paper */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Artifacts</h3>
          <ArtifactList experimentId={id!} />
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Model</h3>
          <DownloadModelButton experimentId={id!} />
        </Card>
        <Card>
          <h3 className="text-sm font-semibold text-white mb-3">Paper</h3>
          <DownloadPaperButton experimentId={id!} />
        </Card>
      </div>
    </div>
  )
}