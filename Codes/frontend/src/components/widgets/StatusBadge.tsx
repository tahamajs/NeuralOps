// src/components/widgets/StatusBadge.tsx
import { Badge } from '../ui/Badge'

export const StatusBadge = ({ status }: { status: string }) => <Badge status={status}>{status}</Badge>