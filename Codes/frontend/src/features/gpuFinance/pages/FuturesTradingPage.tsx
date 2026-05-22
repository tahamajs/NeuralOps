// src/features/gpuFinance/pages/FuturesTradingPage.tsx
import { useState, useEffect } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { gpuFinanceAPI } from '../services/api'
import { useWebSocketPrices } from '../hooks/useWebSocketPrices'
import { OrderBook } from '../components/OrderBook'
import { PositionCard } from '../components/PositionCard'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Input } from '../../../components/ui/Input'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { DataGrid } from '../../../components/ui/DataGrid'
import { TrendingUp, TrendingDown, Play, XCircle, Activity } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function FuturesTradingPage({ onClose }: { onClose?: () => void }) {
  const [selectedContract, setSelectedContract] = useState<string>('')
  const [side, setSide] = useState<'buy' | 'sell'>('buy')
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market')
  const [quantity, setQuantity] = useState(1)
  const [price, setPrice] = useState(0)
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data: contracts, isLoading: contractsLoading } = useQuery({
    queryKey: ['futures-contracts'],
    queryFn: () => gpuFinanceAPI.getContracts(),
  })

  const { data: orders, isLoading: ordersLoading } = useQuery({
    queryKey: ['futures-orders'],
    queryFn: () => gpuFinanceAPI.getOrders(),
    refetchInterval: 5000,
  })

  const { data: positions } = useQuery({
    queryKey: ['futures-positions'],
    queryFn: () => gpuFinanceAPI.getPositions(),
    refetchInterval: 5000,
  })

  const { prices, isConnected } = useWebSocketPrices(selectedContract)

  const placeOrder = useMutation({
    mutationFn: () => gpuFinanceAPI.placeOrder({ contract_id: selectedContract, side, type: orderType, quantity, price: orderType === 'limit' ? price : undefined }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['futures-orders'] })
      queryClient.invalidateQueries({ queryKey: ['futures-positions'] })
      success('Order placed')
    },
    onError: () => error('Order failed'),
  })

  const cancelOrder = useMutation({
    mutationFn: (id: string) => gpuFinanceAPI.cancelOrder(id),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['futures-orders'] }),
  })

  const currentPrice = prices?.current || 0
  const selectedContractObj = contracts?.find(c => c.id === selectedContract)

  if (contractsLoading || ordersLoading) return <Spinner />

  const orderColumns = [
    { accessorKey: 'id', header: 'ID', cell: ({ row }: any) => row.original.id.slice(0, 8) },
    { accessorKey: 'side', header: 'Side', cell: ({ row }: any) => <span className={row.original.side === 'buy' ? 'text-green-400' : 'text-red-400'}>{row.original.side}</span> },
    { accessorKey: 'type', header: 'Type' },
    { accessorKey: 'quantity', header: 'Qty' },
    { accessorKey: 'filled_quantity', header: 'Filled' },
    { accessorKey: 'price', header: 'Price', cell: ({ row }: any) => row.original.price?.toFixed(4) || 'Market' },
    { accessorKey: 'status', header: 'Status' },
    { id: 'actions', header: '', cell: ({ row }: any) => row.original.status === 'open' && <button onClick={() => cancelOrder.mutate(row.original.id)}><XCircle size={14} className="text-red-400" /></button> },
  ]

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><TrendingUp size={20} className="text-neon-mint" /> Futures Trading</h1><p className="text-xs text-slate-400">GPU compute futures contracts</p></div>
        <div className="flex gap-2"><div className={`px-2 py-1 rounded text-xs ${isConnected ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'}`}>{isConnected ? '● Live' : '○ Offline'}</div>{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <TerminalCard className="p-4 space-y-3">
            <Select label="Contract" options={contracts?.map(c => ({ value: c.id, label: `${c.underlying_index} (expires ${new Date(c.expiry_date).toLocaleDateString()})` })) || []} value={selectedContract} onChange={e => setSelectedContract(e.target.value)} />
            <div className="grid grid-cols-2 gap-2">
              <div className="flex gap-1"><button onClick={() => setSide('buy')} className={`flex-1 py-2 rounded ${side === 'buy' ? 'bg-green-500/30 text-green-400' : 'bg-white/5'}`}>Buy</button><button onClick={() => setSide('sell')} className={`flex-1 py-2 rounded ${side === 'sell' ? 'bg-red-500/30 text-red-400' : 'bg-white/5'}`}>Sell</button></div>
              <Select options={[{ value: 'market', label: 'Market' }, { value: 'limit', label: 'Limit' }]} value={orderType} onChange={e => setOrderType(e.target.value as any)} />
            </div>
            <Input label="Quantity (contracts)" type="number" value={quantity} onChange={e => setQuantity(parseInt(e.target.value))} />
            {orderType === 'limit' && <Input label="Limit Price" type="number" step="0.01" value={price} onChange={e => setPrice(parseFloat(e.target.value))} />}
            <div className="text-sm text-slate-400">Current price: ${currentPrice.toFixed(4)}</div>
            <Button onClick={() => placeOrder.mutate()} disabled={!selectedContract || quantity <= 0 || (orderType === 'limit' && price <= 0)} className="w-full"><Play size={14} className="mr-1" /> Place Order</Button>
          </TerminalCard>
          <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Open Positions</h3>{positions?.length ? positions.map(p => <PositionCard key={p.contract_id} position={p} contract={contracts?.find(c => c.id === p.contract_id)} />) : <div className="text-slate-500 text-sm">No open positions</div>}</TerminalCard>
        </div>
        <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Open Orders</h3><DataGrid data={orders?.filter(o => o.status === 'open') || []} columns={orderColumns} /></TerminalCard>
        <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Order History</h3><DataGrid data={orders?.filter(o => o.status !== 'open') || []} columns={orderColumns} /></TerminalCard>
      </div>
    </div>
  )
}