import { useQuery, useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import api from '../../../api/client'

export default function MyOrdersPage() {
  const { data, refetch } = useQuery({
    queryKey: ['my-orders'],
    queryFn: () => api.get('/marketplace/orders/').then(r => r.data),
  })

  const createOrder = useMutation({
    mutationFn: (order: any) => api.post('/marketplace/orders/', order),
    onSuccess: () => refetch(),
  })

  const [newOrder, setNewOrder] = useState({
    order_type: 'buy',
    order_mode: 'market',
    gpu_model_preference: '',
    min_vram_gb: '',
    max_price_per_hour: '',
    required_hours: 1,
    docker_image: '',
    command: '',
  })

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">My Orders</h1>

      <div className="glass p-4 mb-6">
        <h2 className="font-bold mb-2">Place New Order</h2>
        <div className="grid grid-cols-2 gap-3">
          <input placeholder="GPU model" value={newOrder.gpu_model_preference} onChange={e => setNewOrder(o => ({...o, gpu_model_preference: e.target.value}))} className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-sm" />
          <input placeholder="Min VRAM" type="number" value={newOrder.min_vram_gb} onChange={e => setNewOrder(o => ({...o, min_vram_gb: e.target.value}))} className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-sm" />
          <input placeholder="Max price/hr" type="number" step="0.01" value={newOrder.max_price_per_hour} onChange={e => setNewOrder(o => ({...o, max_price_per_hour: e.target.value}))} className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-sm" />
          <input placeholder="Hours" type="number" value={newOrder.required_hours} onChange={e => setNewOrder(o => ({...o, required_hours: Number(e.target.value)}))} className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-sm" />
        </div>
        <button
          onClick={() => createOrder.mutate(newOrder)}
          className="mt-3 bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded-lg text-sm font-semibold transition-colors"
          disabled={createOrder.isLoading}
        >
          {createOrder.isLoading ? 'Placing...' : 'Place Order'}
        </button>
      </div>

      <div className="space-y-2">
        {data?.map((order: any) => (
          <div key={order.id} className="glass p-3 flex justify-between">
            <div>
              <span className="font-bold">{order.gpu_model_preference || 'Any'}</span>
              <span className="text-slate-400 text-xs ml-2">· {order.order_mode} · {order.status}</span>
            </div>
            <span className="text-sm text-slate-400">{new Date(order.created_at).toLocaleString()}</span>
          </div>
        ))}
        {data?.length === 0 && <p className="text-slate-500">No orders yet.</p>}
      </div>
    </div>
  )
}