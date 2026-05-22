import { Badge } from '../../../components/ui/Badge'

const stageConfig = {
  development: { label: 'Dev', color: 'secondary' },
  staging: { label: 'Staging', color: 'info' },
  production: { label: 'Production', color: 'success' },
  archived: { label: 'Archived', color: 'danger' },
}

export function StageBadge({ stage }: { stage: string }) {
  const config = stageConfig[stage as keyof typeof stageConfig] || stageConfig.development
  return <Badge variant={config.color as any}>{config.label}</Badge>
}