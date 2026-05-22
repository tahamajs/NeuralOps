import { cn } from '../../lib/utils'
import { cva, type VariantProps } from 'class-variance-authority'

const badgeVariants = cva(
  'inline-flex items-center px-2.5 py-0.5 rounded-full text-[10px] font-bold uppercase tracking-wider border',
  {
    variants: {
      variant: {
        queued: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
        running: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
        completed: 'bg-green-500/10 text-green-400 border-green-500/30',
        failed: 'bg-red-500/10 text-red-400 border-red-500/30',
        stopped: 'bg-gray-500/10 text-gray-400 border-gray-500/30',
        pending: 'bg-yellow-500/10 text-yellow-400 border-yellow-500/30',
        active: 'bg-blue-500/10 text-blue-400 border-blue-500/30',
        paused: 'bg-gray-500/10 text-gray-400 border-gray-500/30',
        neutral: 'bg-white/5 text-slate-400 border-white/10',
        success: 'bg-green-500/10 text-green-400 border-green-500/30',
        warning: 'bg-orange-500/10 text-orange-400 border-orange-500/30',
      },
    },
    defaultVariants: { variant: 'neutral' },
  }
)

interface BadgeProps extends VariantProps<typeof badgeVariants> {
  children?: React.ReactNode
  className?: string
  dot?: boolean
}

export const Badge = ({ variant, children, className, dot }: BadgeProps) => (
  <span className={cn(badgeVariants({ variant }), className)}>
    {dot && <span className={cn('w-1.5 h-1.5 rounded-full mr-1', variant === 'running' && 'animate-pulse bg-blue-400')} />}
    {children || variant}
  </span>
)