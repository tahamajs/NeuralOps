// This is a thin wrapper – you can just use Badge directly, but kept for compatibility
import { Badge } from '../ui/Badge'

export const StatusBadge = ({ status }: { status: string }) => <Badge status={status}>{status}</Badge>