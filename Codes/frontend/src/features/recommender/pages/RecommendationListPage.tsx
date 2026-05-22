import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { recommenderAPI } from '../services/api'
import { RecommendationCard } from '../components/RecommendationCard'
import { OpportunityIndexWidget } from '../components/OpportunityIndexWidget'
import { TerminalCard } from '../../../components/ui/TerminalCard'
import { Button } from '../../../components/ui/Button'
import { Spinner } from '../../../components/ui/Spinner'
import { Tabs, TabList, Tab, TabPanel } from '../../../components/ui/Tabs'
import { Lightbulb, RefreshCw } from 'lucide-react'
import { useToast } from '../../../components/ui/Toast'

export default function RecommendationListPage({ onClose }: { onClose?: () => void }) {
  const [typeFilter, setTypeFilter] = useState('')
  const queryClient = useQueryClient()
  const { success, error } = useToast()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['recommendations', { type: typeFilter }],
    queryFn: () => recommenderAPI.getRecommendations({ type: typeFilter || undefined }),
  })

  const regenerate = useMutation({
    mutationFn: () => recommenderAPI.regenerateAll(),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['recommendations'] })
      success('Recommendations regenerated')
    },
    onError: () => error('Failed to regenerate'),
  })

  const recommendations = data?.results || []

  return (
    <div className="flex flex-col h-full">
      <div className="flex justify-between items-center p-4 border-b border-white/10">
        <div><h1 className="text-xl font-bold text-white">Experiment Recommendations</h1><p className="text-xs text-slate-400">AI‑powered suggestions to boost your research</p></div>
        <div className="flex gap-2">
          <Button size="sm" variant="ghost" onClick={() => regenerate.mutate()}><RefreshCw size={14} /> Regenerate</Button>
          {onClose && <Button size="sm" variant="ghost" onClick={onClose}>✕</Button>}
        </div>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4">
        <OpportunityIndexWidget />
        <div className="flex gap-2 overflow-x-auto pb-2">
          <button onClick={() => setTypeFilter('')} className={`px-3 py-1 rounded-full text-xs font-medium ${!typeFilter ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300'}`}>All</button>
          <button onClick={() => setTypeFilter('hyperparameter')} className={`px-3 py-1 rounded-full text-xs font-medium ${typeFilter === 'hyperparameter' ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300'}`}>Hyperparameters</button>
          <button onClick={() => setTypeFilter('architecture')} className={`px-3 py-1 rounded-full text-xs font-medium ${typeFilter === 'architecture' ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300'}`}>Architecture</button>
          <button onClick={() => setTypeFilter('data_augmentation')} className={`px-3 py-1 rounded-full text-xs font-medium ${typeFilter === 'data_augmentation' ? 'bg-neon-blue text-white' : 'bg-white/10 text-slate-300'}`}>Data Augmentation</button>
        </div>
        {isLoading ? <Spinner /> : recommendations.length === 0 ? (
          <TerminalCard className="p-8 text-center"><Lightbulb size={32} className="mx-auto mb-2 text-slate-500" /><p className="text-slate-400">No recommendations yet. Run more experiments to get AI‑powered suggestions.</p></TerminalCard>
        ) : (
          <div className="grid md:grid-cols-2 gap-4">
            {recommendations.map(rec => <RecommendationCard key={rec.id} recommendation={rec} />)}
          </div>
        )}
      </div>
    </div>
  )
}
