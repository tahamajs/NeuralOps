import { cn } from '../../lib/utils'

export const Table = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <div className="overflow-x-auto">
    <table className={cn('w-full text-xs', className)}>{children}</table>
  </div>
)

export const THead = ({ children }: { children: React.ReactNode }) => (
  <thead>
    <tr className="text-left text-slate-500 font-black uppercase tracking-[0.2em] border-b border-white/5">
      {children}
    </tr>
  </thead>
)

export const Th = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <th className={cn('px-6 py-4', className)}>{children}</th>
)

export const TBody = ({ children }: { children: React.ReactNode }) => (
  <tbody className="divide-y divide-white/5">{children}</tbody>
)

export const Tr = ({ children, className, onClick }: { children: React.ReactNode; className?: string; onClick?: () => void }) => (
  <tr
    className={cn('group hover:bg-white/[0.02] transition-colors', onClick && 'cursor-pointer', className)}
    onClick={onClick}
  >
    {children}
  </tr>
)

export const Td = ({ children, className }: { children: React.ReactNode; className?: string }) => (
  <td className={cn('px-6 py-4', className)}>{children}</td>
)