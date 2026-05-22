import { cn } from '../../lib/utils'

interface SpinnerProps {
  className?: string
  size?: 'sm' | 'md' | 'lg'
}

const sizeMap = { sm: 'w-4 h-4', md: 'w-8 h-8', lg: 'w-12 h-12' }

export const Spinner = ({ className, size = 'md' }: SpinnerProps) => (
  <div className={cn('flex items-center justify-center', className)}>
    <div
      className={cn(
        'border-2 border-white/10 border-t-neon-blue rounded-full animate-spin',
        sizeMap[size]
      )}
    />
  </div>
)