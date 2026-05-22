import { useQuery, useMutation } from '@tanstack/react-query'
import api from '../../api/client'
import { Card, Button, Spinner, Table } from '../../components/ui'
import { CreditCard, Download } from 'lucide-react'

export function OrganizationBilling({ orgId }: { orgId: string }) {
  const { data: subscription, isLoading: subLoading } = useQuery({
    queryKey: ['org-subscription', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/subscription/`)).data,
  })

  const { data: plans } = useQuery({
    queryKey: ['plans'],
    queryFn: async () => (await api.get('/plans/')).data,
  })

  const { data: invoices } = useQuery({
    queryKey: ['org-invoices', orgId],
    queryFn: async () => (await api.get(`/orgs/${orgId}/invoices/`)).data,
  })

  const changePlan = useMutation({
    mutationFn: (planId: string) => api.post(`/orgs/${orgId}/subscription/change/${planId}/`, {}),
    onSuccess: () => window.location.reload(),
  })

  if (subLoading) return <Spinner />

  return (
    <div className="space-y-6">
      {/* Current Plan */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-2">Current Plan</h3>
        <div className="flex justify-between items-center">
          <div>
            <p className="text-xl font-bold text-white">{subscription?.plan_name || 'Free'}</p>
            <p className="text-sm text-slate-400">
              {subscription?.status === 'active' ? 'Active' : 'Inactive'}
            </p>
          </div>
          <div className="text-right">
            <p className="text-2xl font-bold text-neon-blue">
              ${subscription?.monthly_price || 0}/month
            </p>
          </div>
        </div>
      </Card>

      {/* Available Plans */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Upgrade Your Plan</h3>
        <div className="grid md:grid-cols-3 gap-4">
          {plans?.map((plan: any) => (
            <div key={plan.id} className="bg-white/5 rounded-xl p-4">
              <h4 className="font-bold text-white">{plan.name}</h4>
              <p className="text-2xl font-bold text-neon-blue mt-2">${plan.price_monthly}<span className="text-sm text-slate-400">/mo</span></p>
              <ul className="mt-3 space-y-1 text-sm text-slate-300">
                {plan.features?.map((f: string) => <li key={f}>✓ {f}</li>)}
              </ul>
              <Button size="sm" className="mt-4 w-full" onClick={() => changePlan.mutate(plan.id)}>Upgrade</Button>
            </div>
          ))}
        </div>
      </Card>

      {/* Invoices */}
      <Card>
        <h3 className="text-sm font-bold text-white mb-3">Billing History</h3>
        <Table>
          <thead><tr><th>Date</th><th>Amount</th><th>Status</th><th></th></tr></thead>
          <tbody>
            {invoices?.map((inv: any) => (
              <tr key={inv.id}>
                <td>{new Date(inv.period_start).toLocaleDateString()}</td>
                <td>${inv.amount}</td>
                <td><span className={`text-xs ${inv.status === 'paid' ? 'text-green-400' : 'text-yellow-400'}`}>{inv.status}</span></td>
                <td><a href={inv.pdf_url} target="_blank"><Download size={14} className="text-neon-blue" /></a></td>
              </tr>
            ))}
          </tbody>
        </Table>
      </Card>
    </div>
  )
}