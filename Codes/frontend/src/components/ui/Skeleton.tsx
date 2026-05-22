import { cn } from '../../lib/utils'

export const Skeleton = ({ className, animated = true }: { className?: string; animated?: boolean }) => (
  <div className={cn('rounded bg-white/10', animated && 'shimmer', className)} />
)

// Grid skeleton for cards
export const CardSkeleton = () => (
  <div className="glass p-6 space-y-4">
    <Skeleton className="h-6 w-3/4" />
    <Skeleton className="h-4 w-full" />
    <Skeleton className="h-4 w-5/6" />
    <div className="flex gap-2">
      <Skeleton className="h-8 w-20 rounded-full" />
      <Skeleton className="h-8 w-24 rounded-full" />
    </div>
  </div>
)