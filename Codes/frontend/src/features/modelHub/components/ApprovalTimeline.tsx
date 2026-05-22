import { Timeline } from 'antd' // or custom – we'll implement a simple one
import { CheckCircle, XCircle, MessageCircle } from 'lucide-react'

export function ApprovalTimeline({ comments }: { comments: ApprovalComment[] }) {
  return (
    <div className="space-y-3">
      {comments.map(c => (
        <div key={c.id} className="flex gap-3 border-l-2 border-white/10 pl-3">
          <div className="shrink-0">
            {c.action === 'approve' && <CheckCircle size={16} className="text-green-400" />}
            {c.action === 'reject' && <XCircle size={16} className="text-red-400" />}
            {c.action === 'request_changes' && <MessageCircle size={16} className="text-yellow-400" />}
          </div>
          <div>
            <div className="text-sm text-white"><span className="font-medium">{c.user}</span> {c.action}</div>
            <div className="text-xs text-slate-400">{new Date(c.created_at).toLocaleString()}</div>
            {c.comment && <div className="text-sm text-slate-300 mt-1">{c.comment}</div>}
          </div>
        </div>
      ))}
    </div>
  )
}