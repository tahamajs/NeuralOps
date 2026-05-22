// src/features/marketplace/components/AdvancedOrderBook.tsx
import { useEffect, useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { marketplaceAPI } from '../services/api'
import { useOrderBookWebSocket } from '../hooks/useOrderBookWebSocket'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Spinner } from '../../../components/ui/Spinner'
import { TrendingUp, TrendingDown } from 'lucide-react'

interface AdvancedOrderBookProps {
  listingId: string
}

export function AdvancedOrderBook({ listingId }: AdvancedOrderBookProps) {
  const [view, setView] = useState<'depth' | 'table'>('table')
  const { data: depth, isLoading, refetch } = useQuery({
    queryKey: ['order-book-depth', listingId],
    queryFn: () => marketplaceAPI.getOrderBookDepth(listingId),
    refetchInterval: 5000,
  })
  const { liveDepth, isConnected } = useOrderBookWebSocket(listingId)
  const currentDepth = liveDepth || depth

  if (isLoading) return <Spinner size="sm" />

  const maxTotal = Math.max(
    ...(currentDepth?.bids.map(b => b.total) || []),
    ...(currentDepth?.asks.map(a => a.total) || [])
  )

  return (
    <TerminalCard className="p-4">
      <div className="flex justify-between items-center mb-3">
        <div className="flex gap-2"><button onClick={() => setView('table')} className={`px-2 py-1 rounded text-xs ${view === 'table' ? 'bg-neon-blue text-white' : 'text-slate-400'}`}>Order Book</button><button onClick={() => setView('depth')} className={`px-2 py-1 rounded text-xs ${view === 'depth' ? 'bg-neon-blue text-white' : 'text-slate-400'}`}>Depth Chart</button></div>
        <div className="text-xs"><span className={isConnected ? 'text-green-400' : 'text-red-400'}>{isConnected ? '● Live' : '○ Polling'}</span></div>
      </div>
      {view === 'table' ? (
        <div className="grid grid-cols-2 gap-4">
          <div><div className="text-xs text-green-400 mb-1">Bids (Buy)</div><div className="space-y-0.5">{currentDepth?.bids.slice(0, 10).map((b, i) => (<div key={i} className="flex justify-between text-sm"><span className="text-white">{b.price.toFixed(4)}</span><span className="text-slate-400">{b.quantity}</span><span className="text-slate-500">{b.total.toFixed(0)}</span></div>))}</div></div>
          <div><div className="text-xs text-red-400 mb-1">Asks (Sell)</div><div className="space-y-0.5">{currentDepth?.asks.slice(0, 10).map((a, i) => (<div key={i} className="flex justify-between text-sm"><span className="text-white">{a.price.toFixed(4)}</span><span className="text-slate-400">{a.quantity}</span><span className="text-slate-500">{a.total.toFixed(0)}</span></div>))}</div></div>
        </div>
      ) : (
        <div className="relative h-48">
          <div className="absolute inset-0">
            {currentDepth?.bids.map(b => (<div key={b.price} className="absolute bottom-0 bg-green-500/20 border border-green-500/50" style={{ height: `${(b.total / maxTotal) * 100}%`, width: '5px', left: `${(b.price / (currentDepth.bids[0]?.price || 1)) * 100}px` }} />))}
          </div>
          <div className="text-center text-slate-500 text-sm">Depth chart placeholder – integrate with a charting library (e.g., lightweight‑charts)</div>
        </div>
      )}
      <div className="mt-2 flex justify-between text-xs text-slate-500"><span>Spread: {currentDepth?.spread?.toFixed(4)}</span><span>Last Price: {currentDepth?.last_price?.toFixed(4)}</span></div>
    </TerminalCard>
  )
}