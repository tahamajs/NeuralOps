import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Check, Zap } from 'lucide-react'

export function PlanCard({ plan, interval, isCurrent, onSubscribe, onCancel }: any) {
  const price = interval === 'monthly' ? plan.price_monthly : plan.price_yearly
  const period = interval === 'monthly' ? '/month' : '/year'
  return (
    <TerminalCard className={`p-4 ${plan.tier === 'enterprise' ? 'border-neon-mint' : ''}`}>
      {plan.tier === 'enterprise' && <Badge variant="success" className="absolute -top-2 right-4">Most Popular</Badge>}
      <h3 className="text-lg font-bold text-white">{plan.name}</h3>
      <div className="mt-2"><span className="text-3xl font-bold text-neon-mint">${price}</span><span className="text-slate-400">{period}</span></div>
      <ul className="mt-4 space-y-2 text-sm">{plan.features.map((f: string) => <li key={f} className="flex items-center gap-2"><Check size={14} className="text-green-400" />{f}</li>)}</ul>
      {isCurrent ? <Button variant="secondary" disabled className="w-full mt-4">Current Plan</Button> : <Button onClick={onSubscribe} className="w-full mt-4">Upgrade</Button>}
      {isCurrent && plan.tier !== 'free' && <Button variant="ghost" onClick={onCancel} className="w-full mt-2">Cancel Subscription</Button>}
    </TerminalCard>
  )
}