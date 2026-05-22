import { Badge } from '../../../components/ui/Badge'

export function QualityScoreBadge({ score }: { score?: number }) {
  if (!score) return null
  const variant = score > 0.9 ? 'success' : score > 0.7 ? 'warning' : 'danger'
  return <Badge variant={variant}>{(score * 100).toFixed(0)}%</Badge>
}