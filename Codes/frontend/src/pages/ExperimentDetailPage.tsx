import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { useWebSocket } from '../hooks/useWebSocket'
import { useLiveStore } from '../store/experimentStore'
import api from '../api/client'
import StatusBadge from '../components/StatusBadge'
import ProgressBar from '../components/ProgressBar'
import LiveLossPlot from '../components/LiveLossPlot'
import LiveLogViewer from '../components/LiveLogViewer'
import ConfigSummary from '../components/ConfigSummary'
import ArtifactList from '../components/ArtifactList'
import DownloadModelButton from '../components/DownloadModelButton'

export default function ExperimentDetailPage() {
  const { id } = useParams<{ id: string }>()
  useWebSocket(id)

  const { data: experiment } = useQuery({
    queryKey: ['experiment', id],
    queryFn: async () => (await api.get(`/experiments/${id}/`)).data,
    enabled: !!id,
  })

  const live = useLiveStore()

  if (!experiment) return <div>Loading...</div>

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-2xl font-bold">{experiment.name || 'Experiment'}</h1>
        <StatusBadge status={live.status || experiment.status} />
      </div>

      <ProgressBar progress={live.progress} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <LiveLossPlot metrics={live.metrics} />
        <ConfigSummary config={experiment.config} />
      </div>

      <LiveLogViewer logs={live.logs} />

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <ArtifactList experimentId={id!} />
        <DownloadModelButton experimentId={id!} />
      </div>
    </div>
  )
}