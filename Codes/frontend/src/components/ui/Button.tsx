import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '../../lib/utils';

const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-xl font-semibold transition-all focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed',
  {
    variants: {
      variant: {
        primary: 'bg-neon-blue/90 text-white hover:bg-neon-blue shadow-lg shadow-neon-blue/20',
        secondary: 'bg-white/10 text-slate-300 hover:bg-white/20',
        ghost: 'bg-transparent text-slate-400 hover:text-white hover:bg-white/5',
        danger: 'bg-red-600/80 text-white hover:bg-red-600 shadow-lg shadow-red-500/20',
      },
      size: {
        sm: 'px-3 py-1.5 text-xs',
        md: 'px-5 py-2.5 text-sm',
        lg: 'px-6 py-3 text-base',
      },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof buttonVariants> {}

export const Button = ({ className, variant, size, ...props }: ButtonProps) => (
  <button className={cn(buttonVariants({ variant, size, className }))} {...props} />
);