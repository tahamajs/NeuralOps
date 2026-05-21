import { cn } from '../../lib/utils'

export const Skeleton = ({ className }: { className?: string }) => (
  <div className={cn('animate-pulse bg-white/10 rounded-xl', className)} />
)