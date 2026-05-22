// src/features/gpuFinance/pages/ComputeIndexPage.tsx
import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import { gpuFinanceAPI } from '../services/api'
import { IndexChart } from '../components/IndexChart'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Select } from '../../../components/ui/Select'
import { Spinner } from '../../../components/ui/Spinner'
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react'

export default function ComputeIndexPage({ onClose }: { onClose?: () => void }) {
  const [selectedIndex, setSelectedIndex] = useState<string>('')
  const { data: indices, isLoading } = useQuery({
    queryKey: ['compute-indices'],
    queryFn: () => gpuFinanceAPI.getIndices(),
  })

  const index = indices?.find(i => i.id === selectedIndex)

  if (isLoading) return <Spinner />

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white flex items-center gap-2"><BarChart3 size={20} className="text-neon-mint" /> Compute Index</h1><p className="text-xs text-slate-400">Real‑time GPU compute market indices</p></div>
        <div className="flex gap-2">{onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}</div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <Select options={indices?.map(i => ({ value: i.id, label: i.name })) || []} value={selectedIndex} onChange={e => setSelectedIndex(e.target.value)} placeholder="Select an index" />
        {index && (<><TerminalCard className="p-4"><div className="flex justify-between items-start"><div><div className="text-2xl font-bold text-white">${index.current_value.toFixed(2)}</div><div className="flex items-center gap-1 mt-1"><span className={index.change_24h >= 0 ? 'text-green-400' : 'text-red-400'}>{index.change_24h >= 0 ? '+' : ''}{index.change_24h.toFixed(2)}%</span><span className="text-slate-400">24h change</span></div></div><div className="text-right"><div className="text-sm text-slate-400">Volume 24h</div><div className="text-white">${index.volume_24h.toLocaleString()}</div></div></div></TerminalCard>
          <TerminalCard className="p-4"><IndexChart data={index.history} /></TerminalCard>
          <TerminalCard className="p-4"><h3 className="text-sm font-bold text-white mb-2">Components</h3><div className="space-y-2">{index.components.map(c => (<div key={c.gpu_model} className="flex justify-between items-center"><span>{c.gpu_model}</span><div className="flex items-center gap-4"><span>${c.price.toFixed(4)}/hr</span><span className="text-slate-400">{c.weight}%</span></div></div>))}</div></TerminalCard>
        </>)}
      </div>
    </div>
  )
}