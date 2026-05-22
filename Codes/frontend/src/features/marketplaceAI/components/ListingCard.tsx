import { MarketplaceListing } from '../types'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Badge } from '../../../components/ui/Badge'
import { Cpu, MapPin, Shield, Zap } from 'lucide-react'

interface ListingCardProps {
  listing: MarketplaceListing
  onCheckCompliance: () => void
  onCompare?: () => void
}

export function ListingCard({ listing, onCheckCompliance, onCompare }: ListingCardProps) {
  return (
    <TerminalCard className="p-3"><div className="flex gap-3"><div className="flex-1"><div className="flex items-center justify-between"><div><div className="font-bold text-white">{listing.gpu_specs.model}</div><div className="text-xs text-slate-400 mt-0.5">Provider: {listing.provider}</div></div><Badge variant={listing.status === 'available' ? 'success' : listing.status === 'limited' ? 'warning' : 'secondary'}>{listing.status}</Badge></div>
        <div className="flex items-center gap-4 mt-2 text-sm"><span className="flex items-center gap-1"><Cpu size={14} className="text-neon-blue" />{listing.gpu_specs.vram_gb}GB VRAM</span><span className="flex items-center gap-1"><MapPin size={14} className="text-slate-400" />{listing.region}</span><span className="flex items-center gap-1"><Shield size={14} className="text-green-400" />{listing.reliability_score}%</span></div>
        <div className="flex items-center gap-2 mt-2">{listing.compliance_badges.map(b => <Badge key={b} variant="info">{b}</Badge>)}</div>
        <div className="flex justify-between items-center mt-3"><div><div className="text-2xl font-bold text-neon-mint">${listing.price_per_hour}<span className="text-sm text-slate-400">/hr</span></div>{listing.price_per_hour_spot && <div className="text-xs text-slate-500">Spot: ${listing.price_per_hour_spot}/hr</div>}</div>
          <div className="flex gap-2"><Button size="sm" variant="ghost" onClick={onCheckCompliance}>Compliance</Button>{onCompare && <Button size="sm" variant="ghost" onClick={onCompare}>Compare</Button>}<Button size="sm" variant="primary">Rent</Button></div>
        </div>
      </div>
    </TerminalCard>
  )
}