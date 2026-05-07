import { cn } from '../../lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  hover?: boolean
}

export const Card = ({ children, className, hover = true, ...props }: CardProps) => (
  <div
    className={cn(
      'glass p-6 transition-all duration-300',
      hover && 'hover:bg-white/[0.07] hover:border-white/20',
      className
    )}
    {...props}
  >
    {children}
  </div>
)