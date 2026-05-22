import { useState } from 'react'
import { MarketplaceListing } from '../types'
import { Badge } from '../../../components/ui/Badge'
import { Button } from '../../../components/ui/Button'
import { ChevronDown, ChevronUp, Eye } from 'lucide-react'

interface DynamicComparisonTableProps {
  listings: MarketplaceListing[]
}

export function DynamicComparisonTable({ listings }: DynamicComparisonTableProps) {
  const [highlightDifferences, setHighlightDifferences] = useState(false)
  if (!listings.length) return <div className="text-center text-slate-500 py-8">No listings to compare</div>

  const allFields = ['model', 'vram', 'price', 'region', 'reliability', 'availability', 'compliance']
  const getCellValue = (listing: MarketplaceListing, field: string) => {
    switch (field) {
      case 'model': return listing.gpu_specs.model
      case 'vram': return `${listing.gpu_specs.vram_gb} GB`
      case 'price': return `$${listing.price_per_hour}/hr`
      case 'region': return listing.region
      case 'reliability': return `${listing.reliability_score}%`
      case 'availability': return `${listing.available_units} / ${listing.total_units}`
      case 'compliance': return listing.compliance_badges.join(', ')
      default: return ''
    }
  }

  const isDifferent = (field: string, idx: number) => { const values = listings.map(l => getCellValue(l, field)); return values[0] !== values[idx] }

  return (
    <div className="overflow-x-auto"><table className="w-full text-sm"><thead><tr><th className="p-2 text-left">Feature</th>{listings.map((l, idx) => (<th key={l.id} className="p-2 text-left"><div className="font-bold text-white">{l.gpu_specs.model}</div><div className="text-xs text-slate-400">{l.provider}</div></th>))}</tr></thead>
      <tbody>{allFields.map(field => (<tr key={field} className="border-t border-white/5"><td className="p-2 font-medium text-slate-400 capitalize">{field}</td>{listings.map((l, idx) => (<td key={l.id} className={`p-2 ${highlightDifferences && isDifferent(field, idx) ? 'bg-yellow-500/10 text-yellow-300' : 'text-slate-300'}`}>{getCellValue(l, field)}</td>))}</tr>))}
        <tr className="border-t border-white/5"><td className="p-2 font-medium text-slate-400">Actions</td>{listings.map(l => (<td key={l.id} className="p-2"><Button size="sm" variant="ghost"><Eye size={12} className="mr-1" /> View</Button></td>))}</tr>
      </tbody></table>
      <div className="mt-3 flex justify-end"><label className="flex items-center gap-2 text-xs"><input type="checkbox" checked={highlightDifferences} onChange={() => setHighlightDifferences(v => !v)} /> Highlight differences</label></div>
    </div>
  )
}