import { Inbox } from 'lucide-react'

export const EmptyState = ({ message, action }: { message: string; action?: React.ReactNode }) => (
  <div className="flex flex-col items-center justify-center py-16 text-slate-500">
    <Inbox size={48} className="mb-4 opacity-20" />
    <p className="text-sm">{message}</p>
    {action && <div className="mt-4">{action}</div>}
  </div>
)