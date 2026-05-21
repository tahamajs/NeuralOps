import { useQuery } from '@tanstack/react-query'
import api from '../../../api/client'

export default function BillingPage() {
  const { data: balance } = useQuery({
    queryKey: ['marketplace-balance'],
    queryFn: () => api.get('/marketplace/balance/').then(r => r.data),
  })
  const { data: invoices } = useQuery({
    queryKey: ['marketplace-invoices'],
    queryFn: () => api.get('/marketplace/invoices/').then(r => r.data),
  })

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">Billing</h1>
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Available Balance</p>
          <p className="text-2xl font-bold">${balance?.balance_usd || '0.00'}</p>
        </div>
        <div className="glass p-4">
          <p className="text-sm text-slate-400">Locked (in escrow)</p>
          <p className="text-2xl font-bold">${balance?.locked_balance_usd || '0.00'}</p>
        </div>
      </div>
      <h2 className="font-bold mb-2">Invoices</h2>
      <div className="space-y-2">
        {invoices?.map((inv: any) => (
          <div key={inv.id} className="glass p-3 flex justify-between">
            <span>{inv.invoice_number} · {inv.status}</span>
            <span className="font-bold">${inv.total_usd}</span>
          </div>
        ))}
      </div>
    </div>
  )
}