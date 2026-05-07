import { Card } from '../../../components/ui/Card'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { FlaskConical, Play } from 'lucide-react'
import { useNavigate } from 'react-router-dom'
import { ExperimentConfig } from '../types'

interface Props {
  config: ExperimentConfig
  runId?: string
  isLaunched?: boolean
}

export default function ExperimentPreviewCard({ config, runId, isLaunched }: Props) {
  const navigate = useNavigate()

  return (
    <Card className="border border-neon-blue/30 bg-neon-blue/5">
      <div className="flex items-center gap-2 mb-2">
        <FlaskConical size={18} className="text-neon-blue" />
        <span className="text-sm font-display font-bold text-white">
          {isLaunched ? '🚀 Experiment Launched!' : 'Experiment Configuration'}
        </span>
      </div>
      <div className="text-xs space-y-1 text-slate-300">
        <p>Model: <span className="text-white">{config.model_name}</span></p>
        <p>Task: <span className="text-white">{config.task_name?.toUpperCase()}</span></p>
        <p>Attention: <span className="text-white">{config.attention_type}</span></p>
        <p>Method: <span className="text-white">{config.method}</span></p>
        <p>Epochs: {config.epochs} · Batch: {config.batch_size} · LR: {config.learning_rate}</p>
      </div>
      {isLaunched && runId && (
        <Button size="sm" className="mt-2" onClick={() => navigate(`/experiments/${runId}`)}>
          <Play size={14} className="mr-1" /> Monitor
        </Button>
      )}
    </Card>
  )
}