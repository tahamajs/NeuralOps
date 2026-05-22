import { formatDistanceToNow } from 'date-fns'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Badge } from '../../../components/ui/Badge'
import type { FeedItem } from '../types'

export function FeedItemComponent({ item }: { item: FeedItem }) {
  const activityLabel = { new_publication: '📄 New paper', new_citation: '📌 New citation', trending: '🔥 Trending' }[item.activity_type]
  return (
    <TerminalCard className="p-3"><div className="flex justify-between"><div><div className="text-sm font-medium text-white">{item.paper.title}</div><div className="text-xs text-slate-400">{item.paper.authors.join(', ')} ({item.paper.year})</div></div><Badge variant="info">{activityLabel}</Badge></div><div className="text-xs text-slate-500 mt-2">{formatDistanceToNow(new Date(item.created_at), { addSuffix: true })}</div></TerminalCard>
  )
}