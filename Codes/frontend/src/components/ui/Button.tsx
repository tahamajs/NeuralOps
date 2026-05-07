// components/ui/Button.tsx
import { cva, type VariantProps } from 'class-variance-authority'
const buttonVariants = cva(
  'inline-flex items-center justify-center rounded-xl font-semibold transition-all focus:outline-none',
  {
    variants: {
      variant: {
        primary: 'bg-blue-600 text-white hover:bg-blue-500 shadow-blue-500/20',
        ghost: 'bg-transparent hover:bg-white/5 text-slate-400',
      },
      size: {
        sm: 'px-3 py-1.5 text-xs',
        md: 'px-6 py-2.5 text-sm',
      },
    },
    defaultVariants: { variant: 'primary', size: 'md' },
  }
)