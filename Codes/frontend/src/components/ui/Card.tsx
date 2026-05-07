import { cn } from '../../lib/utils'

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  hover?: boolean
}

export const Card = ({ children, className, hover = true, ...props }: CardProps) => (
  <div
    className={cn(
      'glass p-5 transition-all duration-300',
      hover && 'card-hover-effect',
      className
    )}
    {...props}
  >
    {children}
  </div>
)